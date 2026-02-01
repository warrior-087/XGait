import math
import torch
import torch.nn as nn

from src.Apps.FusionGait.GaitRecognition.IdentityModel.ModelConfigs.model_config import ModelConfig


class BaseModel(nn.Module):
    """
    Optional common interface. Subclasses implement forward().
    Optionally implement get_embedding(x) to extract features.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, x):
        raise NotImplementedError

    def get_embedding(self, x):
        # default: return logits (override in subclass if needed)
        return self.forward(x)


class CNN(BaseModel):
    def __init__(self, md_cfg: ModelConfig = ModelConfig()):
        super().__init__(md_cfg)

        # 1. Expand the feature-extraction part (increase depth and channel count)
        self.feature_extractor = nn.Sequential(
            # Block 1: [B, C, H, W] -> [B, 32, H/2, W/2]
            nn.Conv2d(md_cfg.input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: [B, 64, H/4, W/4] -> [B, 128, H/8, W/8]
            # Adding this layer can significantly improve the model's ability to capture gait details
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # Adaptive pooling keeps the output at 4x4
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 2. Strengthen the fully connected layers
        # Input dimension changes from 32*4*4 to 128*4*4 = 2048
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, md_cfg.embedding_dim),
            nn.BatchNorm1d(md_cfg.embedding_dim),  # Add BN to FC as well for faster convergence
            nn.ReLU(True),
            nn.Dropout(p=md_cfg.dropout),  # Add moderate dropout to improve generalization
            nn.Linear(md_cfg.embedding_dim, md_cfg.num_classes)
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat = self.adaptive_pool(feat)
        feat = torch.flatten(feat, 1)  # Cleaner than view()
        out = self.fc(feat)
        return out


class LateFusionCNN(BaseModel):
    def __init__(self, md_cfg: ModelConfig):
        super().__init__(md_cfg)

        # 1. Define an independent feature extractor (Encoder) for each modality
        # Assume md_cfg.modality_names = ["wifi", "acoustic"]
        self.encoders = nn.ModuleList([
            self._build_encoder(in_c)
            for in_c in md_cfg.modalities_channels
        ])

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 2. Fully-connected layers after fusion
        # Dimension after concatenation: 128 * 4 * 4 * (#modalities)
        fusion_dim = 128 * 4 * 4 * md_cfg.num_modalities

        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, md_cfg.embedding_dim),
            nn.BatchNorm1d(md_cfg.embedding_dim),
            nn.ReLU(True),
            nn.Dropout(p=0.6),  # Fusion models overfit more easily; slightly increase dropout
            nn.Linear(md_cfg.embedding_dim, md_cfg.num_classes)
        )

    @staticmethod
    def _build_encoder(in_channels):
        """Reuse the previous CNN feature-extractor logic."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2),
        )

    def forward(self, x_list):
        """
        x_list: [tensor_wifi, tensor_acoustic]
        """
        feats = []
        # Extract features for each modality separately
        for i, x in enumerate(x_list):
            f = self.encoders[i](x)
            f = self.adaptive_pool(f)
            f = torch.flatten(f, 1)
            feats.append(f)

        # 3. Feature fusion: concatenate along the feature dimension
        merged = torch.cat(feats, dim=1)

        # 4. Classifier
        out = self.fc(merged)
        return out


class ConvLSTMEncoder(nn.Module):
    def __init__(self, in_channels, embedding_dim=128, lstm_hidden=128, dropout_p=0.3):
        super().__init__()

        # Improved CNN: preserve some spatial structure instead of collapsing to (1,1) immediately
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),  # downsample along frequency

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 4))  # keep a local distribution over frequency
        )

        # Input dimension is computed based on AdaptiveAvgPool2d's output
        self.proj = nn.Linear(128 * 1 * 4, embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden,
            num_layers=2,  # increase layers to improve nonlinear capacity
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p if 2 > 1 else 0
        )

        self.dropout = nn.Dropout(dropout_p)
        self.out_dim = lstm_hidden * 2

    def forward(self, x):
        """
        x: [B, C, T, F] -> T is time steps, F is Doppler frequency bins
        """
        B, C, T, F = x.shape

        # Merge B and T so the CNN processes all time steps in parallel on GPU
        # This avoids explicit for-loops and significantly speeds up training
        x = x.transpose(1, 2).contiguous()  # [B, T, C, F]
        x = x.view(B * T, C, 1, F)  # [B*T, C, 1, F]

        ft = self.cnn(x)  # [B*T, 128, 1, 4]
        ft = ft.view(B * T, -1)  # [B*T, 512]
        ft = self.proj(ft)  # [B*T, D]

        feats = ft.view(B, T, -1)  # [B, T, D]
        feats = self.dropout(feats)

        # LSTM processing
        out, (h_n, c_n) = self.lstm(feats)  # out: [B, T, hidden*2]

        # Optimization: for bidirectional LSTM, concatenate the final hidden states of both directions
        # h_n shape: [num_layers * num_directions, B, hidden]
        final_h = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=-1)

        return final_h


class Conv3dLSTM(BaseModel):
    def __init__(self, md_cfg):
        super().__init__(md_cfg)

        self.encoder = ConvLSTMEncoder(
            in_channels=md_cfg.input_channels,
            embedding_dim=md_cfg.embedding_dim,
            lstm_hidden=md_cfg.embedding_dim,
            dropout_p=md_cfg.dropout
        )

        self.fc = nn.Sequential(
            nn.Linear(self.encoder.out_dim, md_cfg.embedding_dim),
            nn.BatchNorm1d(md_cfg.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(md_cfg.dropout),
            nn.Linear(md_cfg.embedding_dim, md_cfg.num_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)
        return self.fc(feat)


class FusionConv3dLSTM(BaseModel):
    def __init__(self, md_cfg):
        super().__init__(md_cfg)

        # 1. Create an independent ConvLSTMEncoder for each modality
        # Even if different modalities have different F (frequency) sizes, AdaptiveAvgPool2d inside the encoder will unify them
        self.streams = nn.ModuleList([
            ConvLSTMEncoder(
                in_channels=md_cfg.input_channels,
                embedding_dim=md_cfg.embedding_dim,
                lstm_hidden=md_cfg.embedding_dim,
                dropout_p=md_cfg.dropout
            ) for _ in range(md_cfg.num_modalities)
        ])

        # 2. Compute the fused dimension
        # Each encoder returns the concatenated bidirectional LSTM state with dimension lstm_hidden * 2
        encoder_out_dim = md_cfg.embedding_dim * 2
        fusion_dim = encoder_out_dim * md_cfg.num_modalities

        # 3. Fusion classifier (Classifier)
        # A slightly deeper classifier often learns nonlinear cross-modality relations better in multimodal tasks
        self.fc = nn.Sequential(
            nn.Linear(fusion_dim, md_cfg.embedding_dim),
            nn.BatchNorm1d(md_cfg.embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(md_cfg.dropout),

            nn.Linear(md_cfg.embedding_dim, md_cfg.embedding_dim // 2),
            nn.BatchNorm1d(md_cfg.embedding_dim // 2),
            nn.ReLU(inplace=True),

            nn.Linear(md_cfg.embedding_dim // 2, md_cfg.num_classes)
        )

    def forward(self, x_list):
        """
        x_list: List[Tensor], e.g. [x_wifi, x_acoustic]
        """
        # Extract temporal features for each modality
        feats = [stream(x) for stream, x in zip(self.streams, x_list)]

        # Feature concatenation (Late Fusion)
        merged = torch.cat(feats, dim=1)

        # Fully-connected classifier
        logits = self.fc(merged)

        # Change here: return only logits to fit the generic trainer
        return logits


class CNNTransformer(BaseModel):
    def __init__(self, md_cfg):
        super().__init__(md_cfg)
        # Unify feature dimension to 256
        self.d_model = 256

        # 1. Improved CNN: reduce information loss
        self.cnn = nn.Sequential(
            # Layer 1: use a smaller stride on the frequency axis to preserve more details
            nn.Conv2d(md_cfg.input_channels, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # No longer collapse to (1,1); keep a (8,1) frequency-structure feature
            nn.AdaptiveAvgPool2d((8, 1))
        )

        # Map [64 * 8] to Transformer d_model [256]
        self.feature_mapping = nn.Linear(64 * 8, self.d_model)

        # 2. Enhanced Transformer Encoder (4 layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=md_cfg.dim_feedforward if hasattr(md_cfg, 'dim_feedforward') else 128,
            dropout=md_cfg.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # 3. Classifier (after inheriting BaseModel, can optionally use md_cfg.num_classes)
        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, md_cfg.num_classes)
        )

    def _add_positional_encoding(self, x):
        batch_size, seq_len, d_model = x.shape
        pe = torch.zeros(seq_len, d_model, device=x.device)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)

    def forward(self, x, return_feat=False):
        # Support list input
        if isinstance(x, (list, tuple)):
            x = x[0]

        B, C, T, F = x.shape
        # Reshape to [Batch*Time, Channel, Freq, 1] and feed into 2D CNN
        x_frames = x.permute(0, 2, 1, 3).reshape(B * T, C, F, 1)

        # CNN feature extraction
        cnn_out = self.cnn(x_frames)  # [B*T, 64, 8, 1]
        cnn_out = cnn_out.view(B, T, -1)  # [B, T, 512]

        # Align dimensions
        feat = self.feature_mapping(cnn_out)  # [B, T, 256]

        # Temporal modeling
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls_tokens, feat), dim=1)
        feat = self._add_positional_encoding(feat)
        feat = self.transformer(feat)

        feat_global = feat[:, 0]  # take CLS token

        if return_feat:
            return feat_global
        return self.fc(feat_global)

class FusionCNNTransformer(BaseModel):
    def __init__(self, md_cfg: ModelConfig):
        super().__init__(md_cfg)

        # Initialize sub-encoders
        self.encoders = nn.ModuleList([
            CNNTransformer(md_cfg) for _ in range(md_cfg.num_modalities)
        ])

        # Input dimension here is hard-coded as 256 * (#modalities)
        # If you want this to be flexible via md_cfg, you can also change 256 in the submodule to md_cfg.embedding_dim
        self.fusion_in_dim = 256 * md_cfg.num_modalities

        self.fc_fusion = nn.Sequential(
            nn.Linear(self.fusion_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(p=0.4),
            nn.Linear(256, md_cfg.num_classes)
        )

    def forward(self, x_list):
        if not isinstance(x_list, (list, tuple)):
            raise ValueError(f"Fusion model expects list, got {type(x_list)}")

        feats = []
        for i, x in enumerate(x_list):
            # Call the submodule and request 256-dim features
            f = self.encoders[i](x, return_feat=True)
            feats.append(f)

        # Concatenate features [B, 512] (assuming two modalities)
        fused_feat = torch.cat(feats, dim=1)

        # Fusion classification
        out = self.fc_fusion(fused_feat)
        return out

# class CNNTransformer(BaseModel):
#     def __init__(self, md_cfg):
#         super().__init__(md_cfg)
#
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))  # learnable classification token
#
#         # 1. Per-frame feature extractor (2D CNN)
#         # Input shape: [B*T, C, F, 1]
#         self.cnn = nn.Sequential(
#             # Layer 1: F -> F/2
#             nn.Conv2d(md_cfg.input_channels, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 1)),
#
#             # Layer 2: F/2 -> F/4
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=(2, 1)),
#
#             # Layer 3: extract higher-level semantics
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1, 1))  # collapse each frame to 128 dims
#         )
#
#         # 2. Transformer Encoder layer
#         # d_model must match the CNN output channels (128)
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=128,
#             nhead=8,
#             dim_feedforward=md_cfg.dim_feedforward,
#             dropout=0.2,
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
#
#         # 3. Classifier
#         # Use parameters from md_cfg
#         embedding_dim = getattr(md_cfg, 'embedding_dim', 256)
#         self.fc = nn.Sequential(
#             nn.Linear(128, embedding_dim),
#             nn.BatchNorm1d(embedding_dim),
#             nn.ReLU(True),
#             nn.Dropout(p=md_cfg.dropout),
#             nn.Linear(embedding_dim, md_cfg.num_classes)
#         )
#
#     def _add_positional_encoding(self, x):
#         """
#         Dynamically generate sinusoidal positional encodings to avoid the prior 1449 > 200 error.
#         """
#         batch_size, seq_len, d_model = x.shape
#         pe = torch.zeros(seq_len, d_model, device=x.device)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#
#         return x + pe.unsqueeze(0)
#
#     def forward(self, x, return_feat=False):
#         # x shape: [B, C, T, F]
#         if isinstance(x, (list, tuple)):
#             x = x[0]
#         B, C, T, F = x.shape  # if the error happens here, x is a list; see Fusion fix below
#         x_frames = x.permute(0, 2, 1, 3).reshape(B * T, C, F, 1)
#         feat = self.cnn(x_frames).view(B, T, 128)
#
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         feat = torch.cat((cls_tokens, feat), dim=1)
#
#         feat = self._add_positional_encoding(feat)
#         feat = self.transformer(feat)
#
#         feat_global = feat[:, 0]  # [B, 128]
#
#         # --- New logic ---
#         if return_feat:
#             return feat_global  # return 128-dim features without passing through self.fc
#
#         out = self.fc(feat_global)
#         return out

# class FusionCNNTransformer(BaseModel):
#     def __init__(self, md_cfg: ModelConfig):
#         super().__init__(md_cfg)
#
#         # Define independent CNN + Transformer for each modality
#         self.encoders = nn.ModuleList([
#             CNNTransformer(md_cfg) for _ in range(md_cfg.num_modalities)
#         ])
#
#         # Final fusion layer after extracting features from all modalities
#         fusion_dim = md_cfg.embedding_dim * md_cfg.num_modalities
#         self.fc_fusion = nn.Sequential(
#             nn.Linear(fusion_dim, md_cfg.embedding_dim),
#             nn.BatchNorm1d(md_cfg.embedding_dim),
#             nn.ReLU(True),
#             nn.Dropout(p=0.4),
#             nn.Linear(md_cfg.embedding_dim, md_cfg.num_classes)
#         )
#
#     def forward(self, x_list):
#         # 1. Robustness check: ensure the input is a list.
#         # If the trainer passes in a Tensor, the multimodal data format adaptation is incorrect
#         if not isinstance(x_list, (list, tuple)):
#             raise ValueError(f"Fusion model expects list input, but got {type(x_list)}")
#
#         feats = []
#         # 2. Iterate over modalities
#         for i, x in enumerate(x_list):
#             # Explicitly tell the sub-model: return features only, not class logits
#             # Also ensure x is a Tensor
#             f = self.encoders[i](x, return_feat=True)
#             feats.append(f)
#
#         # 3. Concatenate features
#         # Each f is [B, 128]; concatenation yields [B, 128 * num_modalities]
#         fused_feat = torch.cat(feats, dim=1)
#
#         # 4. Pass through the fusion head
#         out = self.fc_fusion(fused_feat)
#         return out


# Define a simple residual block
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    """Standard residual-connection block"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Shortcut: if input/output dims differ, use a 1x1 conv to match them
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv_path(x)
        out += self.shortcut(x)  # 🔑 This is the actual ResNet residual addition
        out = F.relu(out)
        return out


class ResNetTransformer(BaseModel):
    def __init__(self, md_cfg):
        super().__init__(md_cfg)
        # CNN backbone: upgraded to a residual structure
        self.stem = nn.Sequential(
            nn.Conv2d(md_cfg.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Replace the original plain conv with a residual block
        self.layer1 = ResidualBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Transformer part stays unchanged
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, batch_first=True, dropout=0.3
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Dimensionality reduction mapped to embedding_dim
        self.feat_fc = nn.Linear(128, md_cfg.embedding_dim)

    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, F = x.shape

        # 1. Speed-up: merge B and T and run CNN once
        # This is much faster than an explicit for-loop
        x = x.permute(0, 2, 1, 3).reshape(B * T, C, 1, F)  # assume F is the frequency dimension

        x = self.stem(x)
        x = self.layer1(x)
        x = self.pool(x)
        x = x.view(B, T, 128)  # restore back to [B, T, 128]

        # 2. Transformer temporal modeling
        x = self.transformer(x)

        # 3. Temporal pooling
        feat = x.mean(dim=1)
        return self.feat_fc(feat)


class FusionResnetTransformer(BaseModel):  # 🔑 inherit BaseModel
    def __init__(self, md_cfg):
        super().__init__(md_cfg)  # call parent init

        self.num_modalities = md_cfg.num_modalities

        # 1. Initialize an independent ResNetTransformer for each modality
        # Assume ResNetTransformer also inherits BaseModel or nn.Module
        self.encoders = nn.ModuleList([
            ResNetTransformer(md_cfg) for _ in range(self.num_modalities)
        ])

        # 2. Feature alignment layers: LayerNorm helps when signal strengths differ a lot across radars
        self.feature_norms = nn.ModuleList([
            nn.LayerNorm(md_cfg.embedding_dim) for _ in range(self.num_modalities)
        ])

        # 3. Modality importance weights: let the model learn which radar signal is more reliable
        self.modality_weights = nn.Parameter(torch.ones(self.num_modalities))

        # 4. Classifier after fusion
        fusion_dim = md_cfg.embedding_dim * self.num_modalities
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),  # stabilize the fused feature distribution
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # high fusion dim -> stronger dropout
            nn.Linear(256, md_cfg.num_classes)
        )

    def forward(self, x_list):
        """
        x_list: can be a tensor (single modality) or a list of tensors (multi-modality)
        """
        # Adapt to trainer input: wrap a single Tensor as a list
        if torch.is_tensor(x_list):
            x_list = [x_list]

        extracted_feats = []

        # Normalize modality weights (Softmax ensures weights sum to 1)
        weights = F.softmax(self.modality_weights, dim=0)

        for i, x in enumerate(x_list):
            # Extract features
            f = self.encoders[i](x)

            # Feature normalization
            f = self.feature_norms[i](f)

            # Apply modality weighting
            f = f * weights[i]

            extracted_feats.append(f)

        # Concatenate all modality features: [B, embedding_dim * num_modalities]
        fused_feat = torch.cat(extracted_feats, dim=1)

        # Final classification
        out = self.classifier(fused_feat)
        return out


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query_feat, key_value_feat):
        # query is modality A, key_value is modality B
        attn_out, _ = self.attn(query_feat, key_value_feat, key_value_feat)
        return self.norm(query_feat + attn_out)


class CNNTransformerWithSelfAttention(BaseModel):
    """
    Improved single-modality version: CNN + Transformer + spatial positional awareness
    """

    def __init__(self, md_cfg):
        super().__init__(md_cfg)

        # 1. Enhanced CNN backbone
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(md_cfg.input_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.AdaptiveAvgPool2d((64, 1))  # 🔑 ensure the sequence length is fixed to 64
        )

        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(md_cfg.input_channels, 32, 3, 1, 1),
        #     nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),  # (32, 62, 75)
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),  # (64, 31, 37)
        #     nn.Conv2d(64, 128, 3, 1, 1),
        #     nn.BatchNorm2d(128), nn.ReLU(True),
        #     # 🔑 key modification: pool only along the frequency axis (W), keep the time axis (H)
        #     nn.AdaptiveAvgPool2d((64, 1))
        # )
        # 2. 🔑 Learnable positional encoding (Position Embedding)
        # Let the model know the relative positions of the 64 tokens on an 8x8 spectrogram
        #self.pos_embedding = nn.Parameter(torch.zeros(1, 64, 128))
        self.pos_embedding = nn.Parameter(torch.zeros(1,64, 128))
        # 3. Standard Transformer Encoder (already contains multi-layer self-attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=8,
            dim_feedforward=512,
            dropout=0.2,  # increase dropout to reduce overfitting
            batch_first=True
        )
        # Increase to 3 layers, enough to capture subtle gait micro-motions
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 4. Classification head: use LayerNorm to stabilize outputs
        self.fc = nn.Sequential(
            nn.Linear(128, md_cfg.embedding_dim),
            nn.LayerNorm(md_cfg.embedding_dim),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(md_cfg.embedding_dim, md_cfg.num_classes)
        )

    def forward(self, x):

        # 1. CNN extracts spatial features [B, 128, 8, 8]
        feat = self.feature_extractor(x)

        # 2. Flatten into a sequence [B, 64, 128]
        #b, c, h, w = feat.shape
        #feat = feat.view(b, c, h * w).permute(0, 2, 1)
        feat = feat.squeeze(-1).permute(0, 2, 1)

        # 3. 🔑 Inject positional information
        feat = feat + self.pos_embedding

        # 4. Transformer processing (self-attention happens inside)
        feat = self.transformer(feat)

        # 5. Global aggregation (sequence mean)
        feat = feat.mean(dim=1)

        # 6. Classification output
        out = self.fc(feat)
        return out

# class CNNTransformerWithSelfAttention(BaseModel):
#     def __init__(self, md_cfg):
#         super().__init__(md_cfg)
#
#         # Assume the input x shape is [B, C, T, 801]
#         # We need to significantly compress the frequency axis (801) while keeping time resolution (T)
#         self.freq_reducer = nn.Sequential(
#             # Step 1: compress the frequency axis (F) with a larger stride
#             nn.Conv2d(md_cfg.input_channels, 32, kernel_size=(3, 7), stride=(1, 4), padding=(1, 3)),
#             nn.BatchNorm2d(32), nn.ReLU(True),
#             nn.Dropout2d(0.4),
#
#             # Step 2: further compress F
#             nn.Conv2d(32, 64, kernel_size=(3, 5), stride=(1, 4), padding=(1, 2)),
#             nn.BatchNorm2d(64), nn.ReLU(True),
#             nn.Dropout2d(0.4),
#
#             # Step 3: use AdaptivePool to fix the frequency axis to a small dimension (e.g., 8)
#             # Then the output becomes [B, 64, T_new, 8]
#             nn.AdaptiveAvgPool2d((None, 8))
#         )
#
#         # Dynamically compute the Transformer input dimension
#         # d_model = 64 (channels) * 8 (remaining freq bins) = 512
#         self.d_model = 64 * 8
#
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=8,
#             dim_feedforward=self.d_model * 2,
#             dropout=0.5,
#             batch_first=True,
#             norm_first=True  # 🔑 must enable, otherwise gradients explode with 10k samples
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)#acoustic 3
#
#         self.fc = nn.Sequential(
#             nn.LayerNorm(self.d_model),
#             nn.Linear(self.d_model, 256),
#             nn.ReLU(True),
#             nn.Linear(256, md_cfg.num_classes)
#         )
#
#     def forward(self, x):
#         # x: [B, C, T, 801]
#
#         # 1. Frequency reduction
#         x = self.freq_reducer(x)  # [B, 64, T_reduced, 8]
#
#         # 2. Reshape to sequence: [B, Time, Features]
#         # Merge 64 channels and 8 frequency bins as the feature vector for each time step
#         b, c, t, f = x.shape
#         x = x.permute(0, 2, 1, 3).contiguous()  # [B, T_reduced, 64, 8]
#         x = x.view(b, t, -1)  # [B, T_reduced, 512]
#
#         # 3. Transformer learns temporal evolution
#         x = self.transformer(x)
#
#         # 4. Pool over time and classify
#         x = x.mean(dim=1)
#         return self.fc(x)


class FusionCNNTransformerWithCrossModalAttention(BaseModel):
    def __init__(self, md_cfg):
        super().__init__(md_cfg)
        self.num_modalities = md_cfg.num_modalities

        # 1. Configure an independent encoder for each modality (following successful single-modality experience)
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(md_cfg.input_channels, 32, 3, 1, 1),
                nn.BatchNorm2d(32), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128), nn.ReLU(True),
                nn.AdaptiveAvgPool2d((8, 8))  # 🔑 unify output to an 8x8 feature map
            ) for _ in range(self.num_modalities)
        ])

        # 2. 🔑 Positional encoding (independent per modality or shared)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 64, 128))

        # 3. Intra-modality Transformer (Self-Attention)
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=128, nhead=8, dim_feedforward=512, dropout=0.3, batch_first=True
                ), num_layers=2
            ) for _ in range(self.num_modalities)
        ])

        # 4. Cross-modality attention (Cross-Attention)
        # Note: query length is 64, key/value length is also 64
        self.cross_modal_attention = CrossModalAttention(embed_dim=128, num_heads=8)

        # 5. Fusion classification head
        self.fc_fusion = nn.Sequential(
            nn.Linear(128 * self.num_modalities, md_cfg.embedding_dim),
            nn.LayerNorm(md_cfg.embedding_dim),  # replace with LayerNorm for more stable training
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(md_cfg.embedding_dim, md_cfg.num_classes)
        )

    def forward(self, x_list):
        # x_list should be a list like [x_acoustic, x_wifi]
        modality_outputs = []

        for i, x in enumerate(x_list):
            # (1) CNN feature extraction -> [B, 128, 8, 8]
            f = self.encoders[i](x)

            # (2) Flatten into a sequence -> [B, 64, 128]
            b, c, h, w = f.shape
            f = f.view(b, c, h * w).permute(0, 2, 1)

            # (3) Inject positional encoding
            f = f + self.pos_embedding

            # (4) Intra-modality self-attention
            f = self.transformers[i](f)
            modality_outputs.append(f)

        # Cross-modality feature fusion
        fused_list = []
        for i, f_q in enumerate(modality_outputs):
            # Use features from "other" modalities as Key/Value
            # For two modalities, "other" is the other one; for multi-modality, concatenate all others
            others = torch.cat([modality_outputs[j] for j in range(len(modality_outputs)) if j != i], dim=1)

            # 🔑 Cross-modality attention: use modality A's query to attend to modality B
            f_cross = self.cross_modal_attention(f_q, others)  # [B, 64, 128]

            # Global aggregation (mean over time/space) -> [B, 128]
            f_gap = f_cross.mean(dim=1)
            fused_list.append(f_gap)

        # Concatenate all fused features -> [B, 128 * num_modalities]
        combined = torch.cat(fused_list, dim=1)

        return self.fc_fusion(combined)
