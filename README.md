# MacV Object Tracker - Interview Assignment

This project demonstrates an **object tracking system** using **YOLOv5** for object detection and **Kalman filters** for smooth and accurate tracking. The solution processes a video file, tracks objects across frames, and exports both a processed video with tracked objects and a CSV file containing detailed tracking data.

## Project Overview

In this assignment, the goal was to:
- Implement an object detection and tracking pipeline using **YOLOv5**.
- Enhance tracking accuracy using **Kalman filters**.
- Provide real-time tracking feedback through a **Flask web application**.
- Generate **CSV reports** detailing object movement across frames.

## Features

- **Real-Time Object Tracking**: Uses YOLOv5 to detect and track objects in a video.
- **Kalman Filter Integration**: Applied to smooth the tracked object's trajectory.
- **Flask Application**: Streams the processed video and displays real-time tracking data on a web interface.
- **Output Files**:
  - **Processed Video**: A video with overlaid bounding boxes and tracking data.
  - **Tracking Data CSV**: A CSV file with object IDs, frame count, and time spent in the video.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/GauravPandit27/MacV-Object-Tracker-Task.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the **YOLOv5 model weights**:
   - Download the model file from [YOLOv5 Releases](https://github.com/ultralytics/yolov5/releases) and place it in the project directory.
   
4. Place your input video in the project directory or specify the correct path in the script.

## Usage

1. Start the Flask server:
    ```bash
    python app.py
    ```

2. Open the application in a browser:
   - Navigate to `http://127.0.0.1:5000` to view the real-time object tracking video stream.

3. The app will process the video, overlay tracking information, and save:
   - **`output_tracked_video.mp4`**: The processed video with object tracking.
   - **`object_tracking_data.csv`**: A CSV file with tracking details for each object.

## Output Details

- **Video**: The processed video will include:
  - Bounding boxes for detected objects.
  - Object ID and real-time position displayed on the video.
  - Trajectory lines showing object movement across frames.
  
- **CSV File**: The CSV will include:
  - **Object ID**: Unique identifier for each object.
  - **Frames Seen**: Total number of frames the object was visible in.
  - **Time in Video**: Duration the object was visible, in seconds.

## Technologies Used

- **Python**: The programming language for the solution.
- **YOLOv5**: Used for real-time object detection.
- **OpenCV**: For video processing and visualizations.
- **Kalman Filter**: For predicting and smoothing the tracked object paths.
- **Flask**: For building the web application to serve real-time video feeds.

## Future Improvements

- **Multi-object tracking**: Currently, the system tracks individual objects but can be extended to handle scenarios with multiple overlapping objects.
- **Model Optimization**: Experimenting with smaller YOLO models or other detectors for efficiency.
- **Real-time Object Recognition**: Integrating classification alongside tracking for broader applications.


*Thank you for reviewing this project!*
