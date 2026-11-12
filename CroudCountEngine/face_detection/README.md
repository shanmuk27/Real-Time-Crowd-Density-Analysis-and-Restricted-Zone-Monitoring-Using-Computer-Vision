# Face-detection training

This trains a face detector; identity matching needs a separate recognition model later.

Download WIDER FACE **Face annotations** and extract its `wider_face_split` folder anywhere inside `C:\Face Validation and detection Dataset`.

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\prepare_face_dataset.py
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\train_face_detector.py
```

The high-resolution YOLO11l training run writes deployment weights to `runs\face_detection\face_yolo11l_1024\weights\best.pt`.

## Live enrolled / unknown face verification

Use the trained YOLOv8n face-detector weights plus ArcFace for identity comparison. Paste the trained
`best.pt` at `CroudCountEngine\\face_detection\\models\\face_yolov8n_best.pt`; this is the default
location, so no `--weights` option is needed. The engine does not save video frames, cropped faces,
CSV files, or enrollment data.

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\face_detection_engine.py
```

The engine asks first for the enrolled-face image/folder location, then for `0` (camera) or `1` (recorded video). A live window marks each detected face as `ENROLLED: name` or `UNKNOWN`; press `q` to stop.

For reliable names, use this enrollment folder layout and include several clear photos per person:

```text
C:\enrolled_faces\
  Alice\photo_1.jpg
  Alice\photo_2.jpg
  Bob\photo_1.jpg
```

The folder names (`Alice`, `Bob`) become the displayed enrolled names. The ArcFace model is downloaded once on its first run and cached as a model dependency; no enrolled faces or camera results are stored by this project.

On a CPU-only PC, the engine automatically uses responsive 512px, non-tiled background inference, a 160px ArcFace crop detector, and verifies at most four new faces per processed frame. On a CUDA NVIDIA PC it automatically uses the full 1024px/tiled accuracy mode.

The live pipeline is low-latency: it keeps at most one unprocessed frame, replaces it with the newest camera frame while inference is busy, and displays each completed frame with the boxes calculated for that exact frame. This prevents boxes from being drawn behind moving faces. The displayed result rate is therefore the real inference rate, while the next inference always starts from a recent camera view rather than from a backlog. Recorded-video input is also paced at its stored frame rate instead of being read from disk as fast as possible.

ArcFace is the expensive part when many faces are visible. On CPU, the engine reuses a matching identity for 5 seconds so it does not repeat ArcFace work before the next slow result is ready; all face-detector boxes are still shown. To prioritize a smoother CPU feed, lower the image size/camera resolution and verification limit, for example:

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\face_detection_engine.py `
  --imgsz 416 --camera-width 640 --camera-height 480 --max-verify-faces 1 `
  --verify-cache-seconds 10 --recognition-det-size 128
```

Use `--max-verify-faces 0` to verify every detected face, or `--verify-cache-seconds 0` to disable identity reuse when maximum per-frame identity freshness matters more than speed.

`last.pt` is saved after every completed epoch. Stop with `Ctrl+C`, then run the same command later. The default `--resume auto` restores the model, optimizer, learning-rate scheduler, EMA state, and original configuration from `last.pt`.

Choose the full epoch target before the first run, e.g. `--epochs 500`. Use `--resume never --run-name another_name` only to start a separate experiment.

## Safe v2 cleanup and fine-tuning

Do not change the v1 dataset while a v1 training run is in progress. The following audit is read-only; it writes reports only:

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\audit_face_dataset.py
```

It produces `generated\face_dataset_audit\dataset_audit.json`, a per-split summary CSV, candidate-image CSV, and an exact-duplicate report. A face is marked tiny-only only when every valid face would be under 8 pixels after normalising its image to 1024 pixels. This is deliberately conservative: hard but valid faces are not automatically treated as noise.

Build the separate v2 dataset after reviewing the audit:

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\prepare_clean_face_dataset_v2.py
```

This creates `generated\face_detection_yolo_v2` without changing WIDER, UFDD, or v1 labels. On the same disk it uses hard links, so it normally needs almost no extra image storage. In its safe default configuration it cleans **only the WIDER training split**: corrupt/missing images, annotations with no usable face boxes, images where every valid face is below the conservative 8-pixel training limit, and the training-side copy of any exact duplicate identified by the audit. Validation and the external UFDD test split remain unchanged so v1/v2 metrics are comparable. It does **not** remove individual tiny boxes from mixed images, avoiding incorrect background labels.

Fine-tune only after copying the v1 `best.pt` from the GPU machine. Start a new run; do not use `--resume` because the dataset changes:

```powershell
.\.venv\Scripts\python.exe .\CroudCountEngine\face_detection\train_face_detector.py `
  --data .\generated\face_detection_yolo_v2\face_dataset_v2.yaml `
  --model D:\path\to\face_yolo11l_1024\weights\best.pt `
  --run-name face_yolo11l_v2_clean --resume auto --epochs 150 --patience 30
```

The new run has early stopping after 30 non-improving validation epochs. If it is interrupted, rerun the same command and `--resume auto` restores its v2 `last.pt`. Keep both v1 and v2 `best.pt` files, then compare them on real restricted-camera images before deployment.

New training runs use roughly half of the available logical CPU cores for image loading and augmentation (`--workers -1`). The neural-network training still runs on the CUDA GPU; moving that work to CPU would make it slower. To reserve even more CPU, specify a lower number such as `--workers 2`.

This machine has no CUDA-enabled PyTorch installation, so the high-resolution run is blocked unless `--allow-cpu` is passed.
