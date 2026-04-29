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
