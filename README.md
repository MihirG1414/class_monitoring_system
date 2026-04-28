# Intelligent Classroom Attention Monitoring System

This repository contains an MVP Streamlit application for classroom attention monitoring. It uses object-oriented Python modules, simulated attention detection, optional YOLO-based phone detection, MediaPipe-based head-pose estimation, SQLite event storage, and CSV report generation.

## Features

- Streamlit dashboard for classroom attention monitoring
- Dummy student seat mapping
- Simulated attentive/distracted detection instead of YOLO or MediaPipe
- Switchable YOLO phone detection mode for uploaded images or videos
- Switchable MediaPipe Face Mesh mode for simple head-down detection
- OOP-based project structure
- SQLite event logging for monitoring history
- CSV report export for dashboard data

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
`-- src/
    |-- __init__.py
    |-- models.py
    |-- monitoring.py
    |-- reporting.py
    |-- simulator.py
    `-- storage.py
```

## Run Locally

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
python -m streamlit run app.py
```

## Demo Login

- Username: `teacher`
- Password: `classroom123`

## Notes

- Simulated mode remains the default workflow for the MVP.
- YOLO mode detects the `cell phone` class only and maps each phone box to the nearest student seat zone.
- MediaPipe mode returns `LOOKING_FORWARD`, `LOOKING_DOWN`, or `FACE_NOT_VISIBLE` per seat.
- A student is only marked distracted when `LOOKING_DOWN` persists for more than 3 seconds in processed video frames.
- YOLO mode processes the first readable video frame to keep phone detection lightweight.
- MediaPipe mode samples video frames over time so the 3-second looking-down threshold can be measured.
- Ultralytics will download the YOLO model weights automatically on the first YOLO run if they are not already cached.
- SQLite data is stored in `data/classroom_events.db`.
- CSV reports are generated in the `reports/` directory.
Project is concluded
