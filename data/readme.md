# XGait Dataset Description

This folder contains the **multi-modality dataset** collected for the fusionGait project. The dataset captures human gait patterns using WiFi CSI (Channel State Information), acoustic signals, and video (for ground truth generation).

## 1. Data Collection Scenarios
XGait is collected in three distinct environments to ensure diversity:
1.  **Laboratory**: A controlled, open environment with minimal furniture interference.
2.  **Home**: A real-world home environment with complex multipath effects caused by furniture and layout.
3.  **Meeting-room**: A small meeting-room environment with central furniture.

<img src="../Tools/Image_res/Scenarios.png" alt="Scenarios" width="800"/>


## 2. Subjects
A total of **27 volunteers** participated in the data collection.
- **Laboratory Subset**: Contains data from **20** subjects.
- **Home Subset**: Contains data from **19** subjects.
- **Meeting-room Subset**: Contains data from **20** subjects.
- **Meeting-room DifferentDays Subset**: Contains data from **6** subjects.
*(Note: Ten subjects participated across three scenarios.)*

## 3. Collection Protocol
Data collection was organized by predefined walking paths in each environment.

| Scenario | Paths | Rounds per Path | Walks per Path | Total Walks per User |
| :--- | :--- | :--- | :--- | :--- |
| **Laboratory** | 9 paths | 30 round trips | 60 walks | 540 samples |
| **Home** | 5 paths | 25 round trips | 50 walks | 250 samples |
| **Meeting-room** | 5 paths | 25 round trips | 50 walks | 250 samples |
| **Meeting-room Different days** | 5 paths | 25-50 round trips |50-100 walks | 250-500 samples |

*Note: "1 Round trip" = 1 walk there + 1 walk back = 2 data instances.*

## 4. Directory Structure
The dataset is organized by **Scenario**, then by **User ID** and **Modality**.

```text
Data/
├── FusionGait/
│   ├── Home/
│   │   ├── senario_cfg.json
│   │   └── user1/
│   │       ├── acoustic/
│   │       │   ├── user1-1-1-r1.wav
│   │       │   ├── user1-1-2-r2.wav
│   │       │   ├── user1-1-3-r3.wav
│   │       │   └── ...
│   │       ├── track_groundTruth/
│   │       │   ├── user1-1-1.npy
│   │       │   ├── user1-1-2.npy
│   │       │   ├── user1-1-3.npy
│   │       │   └── ... 
│   │       └── wifi/
│   │           ├── user1-1-1-r1.dat
│   │           ├── user1-1-2-r1.dat
│   │           ├── user1-1-3-r1.dat
│   │           └── ...
│   ├── Laboratory/
│   │   ├── senario_cfg.json
│   │   └── user1/
│   │       ├── ...
|   ├── MeetingRoom/
│   ├── MeetingRoom_DifferentDays/
└── GaitVideo/
```

**File descriptions**:
- **`wifi/*.dat`**: Raw CSI data from Intel 5300 NICs (Channel State Information)
- **`acoustic/*.wav`**: Audio signals captured by microphones
- **`track_groundTruth/*.npy`**: Ground-truth trajectory data (processed from video)

## 5. Data Download

Due to the large size of the full dataset (100GB+), the XGait dataset is hosted on Kaggle rather than directly in this GitHub repository.

| Platform | Link |
|:---------|:-----|
| **Kaggle** | [XGait-Paper-Dataset](https://www.kaggle.com/datasets/weilaiyilai55/xgait-paper-dataset) |

After downloading the four scenario folders (**Home**, **Laboratory**, **MeetingRoom**, **MeetingRoom-DifferentDays**) from Kaggle, place them under the `data/FusionGait/` directory.

Please refer to the main [README.md](../README.md) for the complete citation and contact information.

## 6. Important Usage Notes and Known Data Issues

### Synchronization and Ground-Truth Alignment

> [!NOTE]
> **The recordings are not strictly synchronized across devices.** Per-frame timestamps were not explicitly recorded, so the two wireless modalities can only be aligned at a coarse level, for example by matching recording intervals and adjusting signal lengths. Equal-length sequences should not be interpreted as precisely time-synchronized observations.
>
> **Video-derived ground truth has a lower temporal resolution than the wireless recordings.** Both the camera frame rate and the resulting trajectory sampling rate are substantially lower than the wireless sampling rates. When evaluating tracking errors, interpolate or resample the reference and predicted trajectories onto a common temporal grid over the corresponding recording interval. This aligns the number of points but does not eliminate timing uncertainty or interpolation error. Please report the alignment procedure used in your experiments.

### Known Missing or Abnormal Data

> [!WARNING]
> Device failures and other acquisition issues resulted in some missing or abnormal data. Check input validity before processing; the collection protocol above describes the intended acquisition schedule, not a guarantee that every file is present and usable.

1. **Unavailable acoustic channel:** Rx2 acoustic data for User3 and Users15–20 are reported as empty/unavailable. Do not treat an empty channel as a valid observation. Here, “empty” refers to unavailable signal data and does not necessarily mean a zero-byte file.

2. **Zero-byte Wi-Fi file:**

   - `Laboratory/user4/wifi/user4-4-7-r1.dat`

3. **Abnormal WAV headers:** The following files failed the WAV header check because the expected RIFF identifier was not found. They require further inspection before use:

   - `Laboratory/user11/acoustic/user11-1-5-r2.wav`
   - `Laboratory/user16/acoustic/user16-6-59-r2.wav`
   - `Laboratory/user17/acoustic/user17-8-14-r2.wav`
   - `Home/user9/acoustic/user9-2-40-r1.wav`
   - `Home/user9/acoustic/user9-2-40-r2.wav`
   - `Home/user9/acoustic/user9-2-40-r3.wav`

4. **Day-aware recording identifiers:** In `MeetingRoom_DifferentDays`, User6's path 3, repetition 41 has both `d6` and `d7` versions, covering all three receivers in both modalities. These are distinct day-tagged recordings, not files to deduplicate. Include the day tag in the recording key, e.g., `(scenario, user, path, repetition, day, receiver, modality)`, to prevent one version from overwriting the other during loading.

### Handling Missing or Invalid Inputs

> [!IMPORTANT]
> Configure or adapt the public data-loading code to skip invalid files/recordings or explicitly mask unavailable channels, depending on the method's input requirements. Skipping or masking is not data recovery: zero-filling an unavailable channel does not restore the missing measurement.
>
> Repair a malformed header only when the underlying signal payload and its format can be verified, and preserve the original file. Empty or missing recordings cannot be reconstructed by simply editing a header. If a loader still raises an error, inspect the reported file and adjust the relevant validation or exception handling; automatic handling of every acquisition anomaly is not guaranteed.
>
> For paired-modality experiments, use the intersection of valid recording identifiers. Document any exclusions, repairs, missing-channel handling, and the final sample counts so that results remain reproducible. Missing trajectory ground truth is not necessarily a data defect; such recordings can still support identification evaluation when identity labels and valid signals are available.
