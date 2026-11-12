Real-Time-Crowd-Density-Analysis-and-Restricted-Zone-Monitoring-Using-Computer-Vision

## High-accuracy crowd model

The crowd model's class is named `person`, but it is trained only on the official NWPU-Crowd head-level boxes: one head box represents one person for counting. Do not merge the face-detection dataset into this training set. Its boxes label faces rather than the NWPU head target, and mixing those targets would reduce localization and crowd-count accuracy.

Train the YOLO11 large crowd model on the CUDA-enabled NVIDIA training computer:

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\trainmode.py
```

The default profile uses `yolo11l.pt`, 1024px training, batch size 1 (safe for 8 GB GPUs and the extremely dense NWPU images), up to 300 epochs, and validation early stopping. The completed model is written to `runs\crowd_yolo11l_1024\weights\best.pt`. Use automatic batch sizing only on a high-memory GPU.

Run high-accuracy inference on the NVIDIA device:

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\main.py --device 0
```

The live app automatically chooses that retrained model after it exists, uses CUDA half-precision inference, keeps CPU threads low for capture/display, and uses 1024px overlapping tiles to preserve small, dense people. Use `--no-tiled` only when the camera view is already low resolution and throughput matters more than tiny-person accuracy.

