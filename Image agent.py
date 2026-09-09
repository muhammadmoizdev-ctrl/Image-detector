"""
Object Detection Agent - Image Mode (Multiprocessing Version)
================================================================
Parallel & Distributed Computing (PDC) Course Project

Description:
    This agent takes an image as input and detects objects present
    in it (e.g. chairs, tables, people, etc.).

PDC Concept Applied:
    The image is split into 4 tiles (2x2 grid), and each tile is
    processed in a SEPARATE PROCESS in parallel using Python's
    multiprocessing module. Results from all processes are then
    merged back into the original image's coordinate space.

    A single-process (sequential) baseline is also run for
    comparison, so that the speedup/overhead of parallelization
    can be measured and discussed.

Usage:
    python image_agent.py                  -> opens a file picker
    python image_agent.py path/to/image.jpg -> runs directly on given image
"""

import sys
import time
import cv2
import multiprocessing as mp
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

MODEL_NAME = "yolov8s.pt"  # 's' (small) = better accuracy than 'n' (nano), still reasonably fast
CONFIDENCE_THRESHOLD = 0.4  # Detections below this confidence are discarded
NMS_IOU_THRESHOLD = 0.4     # Boxes overlapping more than this are treated as duplicates

# Each worker process loads its own copy of the model (module-level global)
_model = None


def init_worker():
    """Initializer function: runs once when a worker process starts."""
    global _model
    _model = YOLO(MODEL_NAME)


def detect_on_tile(args):
    """Runs object detection on a single image tile."""
    tile, x_offset, y_offset = args
    results = _model(tile, verbose=False)

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        name = _model.names[class_id]
        conf = float(box.conf[0])

        if conf < CONFIDENCE_THRESHOLD:
            continue  # Skip low-confidence / unreliable detections

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Convert tile-local coordinates back to full-image coordinates
        detections.append({
            "name": name,
            "conf": conf,
            "box": (x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset)
        })
    return detections


def apply_global_nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    """
    Removes duplicate detections that occur when the same object is
    caught by two adjacent tiles (a common side-effect of tiling).
    Keeps the highest-confidence box among overlapping duplicates.
    """
    if not detections:
        return []

    boxes = [d["box"] for d in detections]
    scores = [d["conf"] for d in detections]

    # cv2.dnn.NMSBoxes expects boxes as (x, y, w, h)
    boxes_xywh = [[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2) in boxes]

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, CONFIDENCE_THRESHOLD, iou_threshold)
    if len(indices) == 0:
        return []

    indices = indices.flatten()
    return [detections[i] for i in indices]


def split_into_tiles(image, rows=2, cols=2, overlap_ratio=0.15):
    """
    Splits an image into a rows x cols grid of tiles, with a small
    overlap between neighboring tiles. The overlap ensures that an
    object sitting near a tile boundary still appears WHOLE in at
    least one tile, so it doesn't get cut in half and counted twice.
    """
    h, w = image.shape[:2]
    tile_h, tile_w = h // rows, w // cols
    overlap_h, overlap_w = int(tile_h * overlap_ratio), int(tile_w * overlap_ratio)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            base_y1, base_y2 = r * tile_h, (r + 1) * tile_h if r < rows - 1 else h
            base_x1, base_x2 = c * tile_w, (c + 1) * tile_w if c < cols - 1 else w

            y1 = max(0, base_y1 - overlap_h)
            y2 = min(h, base_y2 + overlap_h)
            x1 = max(0, base_x1 - overlap_w)
            x2 = min(w, base_x2 + overlap_w)

            tile = image[y1:y2, x1:x2]
            tiles.append((tile, x1, y1))
    return tiles


def draw_and_save(image, detections, out_path="output_detected.jpg"):
    """Draws bounding boxes and labels, then saves the annotated image."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = f"{det['name']} {det['conf']*100:.1f}%"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(out_path, image)
    print(f"[SAVED] Annotated result saved to: {out_path}")


def run_multiprocess(image, num_workers=4):
    """Splits the image into tiles and detects objects in parallel."""
    tiles = split_into_tiles(image, rows=2, cols=2)

    start = time.time()
    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.map(detect_on_tile, tiles)
    elapsed = time.time() - start

    all_detections = [d for tile_result in results for d in tile_result]
    all_detections = apply_global_nms(all_detections)
    return all_detections, elapsed


def run_single_process(image):
    """Runs detection on the whole image sequentially (baseline)."""
    init_worker()
    start = time.time()
    detections = detect_on_tile((image, 0, 0))
    elapsed = time.time() - start
    return detections, elapsed


def pick_image_via_dialog():
    """Opens a native file-picker dialog for selecting an image."""
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select an image for detection",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return path


def print_summary(detections):
    """Prints a clean, professional summary of detected objects."""
    if not detections:
        print("\nNo objects detected above the confidence threshold.")
        return

    counts = {}
    for d in detections:
        counts[d["name"]] = counts.get(d["name"], 0) + 1

    print("\n" + "-" * 40)
    print("DETECTION SUMMARY")
    print("-" * 40)
    for name, count in sorted(counts.items()):
        print(f"  {name:<15} x{count}")
    print("-" * 40)

    print("\nDetailed detections:")
    for d in detections:
        print(f"  - {d['name']:<12} confidence: {d['conf']*100:.1f}%")


def main():
    if len(sys.argv) >= 2:
        image_path = sys.argv[1]
    else:
        print("[INFO] Opening file picker... please select an image.")
        image_path = pick_image_via_dialog()

    if not image_path:
        print("[INFO] No image selected. Exiting.")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    print("\n" + "=" * 50)
    print("OBJECT DETECTION AGENT - IMAGE MODE")
    print("=" * 50)

    print("\n[INFO] Running single-process (baseline) detection...")
    single_dets, single_time = run_single_process(image.copy())

    print("[INFO] Running multiprocessing (4 parallel tiles) detection...")
    multi_dets, multi_time = run_multiprocess(image.copy(), num_workers=4)

    print_summary(multi_dets)

    print("\n" + "-" * 40)
    print("PERFORMANCE COMPARISON")
    print("-" * 40)
    print(f"  Single-process time  : {single_time:.3f} sec")
    print(f"  Multiprocessing time : {multi_time:.3f} sec")
    if multi_time > 0:
        print(f"  Speedup factor       : {single_time / multi_time:.2f}x")
    print("-" * 40)

    draw_and_save(image.copy(), multi_dets, out_path="output_detected.jpg")
    print("\n[DONE]\n")


if __name__ == "__main__":
    main()