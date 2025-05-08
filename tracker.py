import cv2
import numpy as np
from collections import defaultdict
import os
import csv
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8s model (you can use yolov8s, yolov8m, etc. depending on your model)
model = YOLO("yolov5m.pt")  # You can also try yolov8m.pt or yolov8l.pt

# Input video path
input_video_path = "/home/gaurav/Desktop/Projects/company/macv-obj-tracking-video.mp4"

# Output CSV path
output_csv_path = "object_tracking_metrics.csv"

# Object tracking storage
object_tracks = defaultdict(list)  # {id: [(x, y), ...]}
object_entry_frame = {}  # {id: first_seen_frame}
object_last_seen_frame = {}  # {id: last_seen_frame}
frame_count = 0

# Initialize Kalman filter dictionary for each object
kalman_filters = {}

# Kalman filter setup function
def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)  # 4 state variables (x, y, dx, dy), 2 measurement variables (x, y)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf

# Load video
cap = cv2.VideoCapture(input_video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))

# Object tracker (YOLOv8)
tracker = model.track(source=input_video_path, stream=True, persist=True, conf=0.4, iou=0.5)

# Process video frame-by-frame
for result in tracker:
    frame = result.orig_img.copy()
    frame_count += 1

    # Display stats on screen
    current_ids = set()

    if result.boxes.id is None:
        continue

    for box in result.boxes:
        obj_id = int(box.id.item())
        cls_id = int(box.cls.item())
        if cls_id != 0:  # Only track 'person' class
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw bounding box and centroid
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Initialize Kalman Filter if not already initialized
        if obj_id not in kalman_filters:
            kalman_filters[obj_id] = create_kalman_filter()
            kalman_filters[obj_id].correct(np.array([cx, cy], np.float32))

        # Kalman prediction
        predicted = kalman_filters[obj_id].predict()
        pred_x, pred_y = int(predicted[0]), int(predicted[1])

        # Update Kalman filter with new measurements
        kalman_filters[obj_id].correct(np.array([cx, cy], np.float32))

        # Draw the predicted position (Kalman Filter smooth prediction)
        cv2.circle(frame, (pred_x, pred_y), 4, (0, 255, 255), -1)

        # Save track
        object_tracks[obj_id].append((cx, cy))
        if obj_id not in object_entry_frame:
            object_entry_frame[obj_id] = frame_count
        object_last_seen_frame[obj_id] = frame_count

        # Draw trail (path traveled by the object)
        pts = object_tracks[obj_id]
        for j in range(1, len(pts)):
            cv2.line(frame, pts[j - 1], pts[j], (255, 0, 0), 2)  # Blue trail

        current_ids.add(obj_id)

    # Show count overlay
    total_unique = len(object_tracks)
    current_count = len(current_ids)
    cv2.putText(frame, f"Current: {current_count} | Total Detected: {total_unique}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Show frame (optional)
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save metrics to CSV
print("\nSaving metrics to CSV...")
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Object ID", "Frames Seen", "Time in Video (s)"])
    for obj_id in object_tracks:
        start = object_entry_frame.get(obj_id, 0)
        end = object_last_seen_frame.get(obj_id, start)
        frames_seen = end - start + 1
        time_seconds = round(frames_seen / fps, 2)
        if time_seconds >= 1.0:  # Filter noise IDs
            writer.writerow([obj_id, frames_seen, time_seconds])

print(f"\nTracking complete. CSV saved to: {output_csv_path}")

