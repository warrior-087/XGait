# constants.py
class ModelNames:
    CNN = "cnn"
    FUSION_CNN = "fusion_cnn"

    LSTM = "lstm"
    FUSION_LSTM = "fusion_lstm"

    TRANSFORMER = "transformer"
    FUSION_TRANSFORMER = "fusion_transformer"

    RESNETTRANSFORMER = "resnetTransformer"
    FUSION_RESNETTRANSFORMER = "fusion_resnetTransformer"

    TRANSFORMERWITHSELFATTENTION = "transformerWithSelfAttention"
    FUSION_TRANSFORMERWITHCROSSMODELATTENTION = "fusion_transformerWithCrossModelAttention"


class Modalities:
    WIFI = "wifi"
    ACOUSTIC = "acoustic"
    FUSION = "fusion"  # "fusion" here does NOT mean late fusion

class TrainerTypes:
    SINGLE = "single"
    FUSION = "fusion"

class Phases:
    TRAIN = "train"
    TEST = "test"
