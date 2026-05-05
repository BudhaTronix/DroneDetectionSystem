# Drone Webcam Detector

## Project Description

Drone Webcam Detector detects drones in real time from a webcam using OpenCV
and a pretrained Ultralytics YOLO drone-detection model. The project downloads
the custom Hugging Face checkpoint `doguilmak/Drone-Detection-YOLOv8x`
(`weight/best.pt`) and saves it as `models/best.pt`.

## Features

- Real-time webcam drone detection
- Pretrained YOLO drone model
- Automatic model download
- Bounding boxes and confidence scores
- Optional annotated video saving
- Conda environment setup
- CLI configuration

## Project Structure

```text
drone-webcam-detector/
  README.md
  requirements.txt
  .gitignore
  src/
    drone_detector/
      __init__.py
      download_model.py
      webcam.py
  models/
    .gitkeep
  runs/
    .gitkeep
```

## Conda Setup

Run commands from the `drone-webcam-detector/` project directory. Create and
activate the dedicated Conda environment named `dds`:

```bash
conda create -n dds python=3.10 -y
conda activate dds
pip install -r requirements.txt
```

All project commands below assume the `dds` environment is activated.

## Download the Pretrained Model

```bash
python -m src.drone_detector.download_model
```

This downloads `weight/best.pt` from `doguilmak/Drone-Detection-YOLOv8x` and
copies it to:

```text
models/best.pt
```

If `models/best.pt` already exists, the download is skipped.

## Run Webcam Detection

```bash
python -m src.drone_detector.webcam
```

## Run With a Different Webcam

```bash
python -m src.drone_detector.webcam --camera 1
```

## Run With a Lower Confidence Threshold

```bash
python -m src.drone_detector.webcam --conf 0.25
```

## Save Annotated Video

```bash
python -m src.drone_detector.webcam --save
```

## Save Annotated Video to a Custom Path

```bash
python -m src.drone_detector.webcam --save --output runs/my_drone_detection.mp4
```

## Run Without Display

```bash
python -m src.drone_detector.webcam --no-display --save
```

## CLI Options

```text
--model models/best.pt
--camera 0
--conf 0.35
--imgsz 640
--device auto
--save
--output runs/drone_webcam_output.mp4
--no-display
```

## Troubleshooting

- Forgetting to activate `dds`: run `conda activate dds`, then reinstall with
  `pip install -r requirements.txt` if needed.
- Missing model file: run `python -m src.drone_detector.download_model`, or let
  `webcam.py` download `models/best.pt` automatically on first run.
- Webcam permissions: allow camera access for your terminal, Python, or IDE in
  your operating system privacy settings.
- Wrong webcam index: try `--camera 1`, `--camera 2`, or another available
  camera index.
- Slow CPU inference: lower `--imgsz`, increase `--conf`, close other apps, or
  use a GPU-capable PyTorch/Ultralytics setup with `--device 0`.
- OpenCV window not opening: check that a desktop display is available. On
  headless systems, use `--no-display --save`.
- Push failing because `origin` is not configured: add a remote first with
  `git remote add origin <REPOSITORY_URL>`, then push with
  `git push origin develop`.

## Git Workflow

All code for this project should be committed and pushed on the `develop`
branch. Do not push project work to `main` or `master`.

```bash
git checkout develop
git add .
git commit -m "<clear commit message>"
git push origin develop
```
