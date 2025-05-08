import cv2
import numpy as np
import csv
from collections import defaultdict
from flask import Flask, Response, render_template_string
from ultralytics import YOLO
import os

app = Flask(__name__)

# Load YOLOv5 model
model = YOLO("yolov5m.pt")

# Input and output video files
video_path = "/home/gaurav/Desktop/Projects/company/macv-obj-tracking-video.mp4"
output_video_path = "output_tracked_video.mp4"
output_csv_path = "object_tracking_data.csv"

# Tracking variables
object_tracks = defaultdict(list)
object_entry_frame = {}
object_last_seen_frame = {}
kalman_filters = {}
frame_count = 0
video_saved = False

def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
    return kf

def save_csv(fps):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Frames Seen", "Time in Video (s)"])
        for obj_id in sorted(object_tracks.keys()):
            frames_seen = object_last_seen_frame[obj_id] - object_entry_frame[obj_id] + 1
            time_seen = round(frames_seen / fps, 2)
            writer.writerow([obj_id, frames_seen, time_seen])

def gen():
    global frame_count, video_saved
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    tracker = model.track(source=video_path, stream=True, persist=True, conf=0.4, iou=0.5)

    for result in tracker:
        frame = result.orig_img.copy()
        frame_count += 1
        if result.boxes.id is None:
            continue

        for box in result.boxes:
            obj_id = int(box.id.item())
            cls_id = int(box.cls.item())
            if cls_id != 0:
                continue  # Only track class 0 (e.g., person)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            if obj_id not in kalman_filters:
                kalman_filters[obj_id] = create_kalman_filter()
                kalman_filters[obj_id].correct(np.array([cx, cy], np.float32))

            predicted = kalman_filters[obj_id].predict()
            pred_x, pred_y = int(predicted[0]), int(predicted[1])
            kalman_filters[obj_id].correct(np.array([cx, cy], np.float32))
            cv2.circle(frame, (pred_x, pred_y), 4, (0, 255, 255), -1)

            object_tracks[obj_id].append((cx, cy))
            if obj_id not in object_entry_frame:
                object_entry_frame[obj_id] = frame_count
            object_last_seen_frame[obj_id] = frame_count

            pts = object_tracks[obj_id]
            for j in range(1, len(pts)):
                cv2.line(frame, pts[j - 1], pts[j], (255, 0, 0), 2)

        out.write(frame)
        _, jpeg = cv2.imencode('.jpg', frame)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    cap.release()
    out.release()
    save_csv(fps)
    video_saved = True

@app.route('/')
def index():
    message = "<script>alert('Video and CSV saved successfully.');</script>" if video_saved else ""
    return render_template_string(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Object Tracking App</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                text-align: center;
                margin: 0;
                padding: 0;
            }}
            h1 {{
                background-color: #333;
                color: #fff;
                padding: 20px 0;
                margin: 0;
            }}
            .video-container {{
                margin-top: 20px;
            }}
            img {{
                border: 2px solid #ccc;
                border-radius: 10px;
            }}
            footer {{
                margin-top: 30px;
                padding: 20px;
                background: #eee;
                color: #444;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <h1>Object Tracking Application</h1>
        <div class="video-container">
            <img src="{{{{ url_for('video_feed') }}}}" width="720" height="540">
        </div>
        <footer>
            <p>This project is developed by Gaurav Pandit. I've been working on a similar initiative for the past 6 months, and it has recently earned me recognition as one of the Top 600 Innovators in Asia by IIT Bombay for my innovative ideas. I believe there is great potential for us to collaborate and achieve exceptional outcomes together. I would love to have a meeting with you to discuss how we can work towards that.

Looking forward to the opportunity.</p>
        </footer>
    </body>
        {message}
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

