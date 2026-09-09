# Image Detection Agent

A parallel & distributed computing project that detects objects in images using YOLOv8 with multiprocessing.

## Features

- **Object Detection**: Detects objects in images using YOLOv8 small model
- **Parallel Processing**: Splits images into 4 tiles (2x2 grid) for parallel processing
- **Performance Comparison**: Compares single-process vs multiprocessing execution time
- **Automatic NMS**: Removes duplicate detections from adjacent tiles

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode (File Picker)
```bash
python "Image agent.py"
```

### Direct Mode (Image Path)
```bash
python "Image agent.py" path/to/your/image.jpg
```

## Configuration

Edit these constants in `Image agent.py`:

- `MODEL_NAME`: YOLOv8 model variant (default: "yolov8s.pt")
- `CONFIDENCE_THRESHOLD`: Minimum detection confidence (default: 0.4)
- `NMS_IOU_THRESHOLD`: NMS overlap threshold (default: 0.4)

## Output

- **Annotated Image**: `output_detected.jpg` with bounding boxes and labels
- **Console Summary**: Detected objects count and confidence scores
- **Performance Metrics**: Execution time comparison

## Performance

The agent compares:
- Single-process baseline detection
- Multiprocessing (4 parallel tiles) detection
- Speedup factor calculation

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch
