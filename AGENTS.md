# AGENTS.md

## Project
Intelligent Classroom Attention Monitoring System with Non-Disruptive Student Alerting.

## Goal
Build a working MVP first. Do not over-engineer. Prioritize a runnable demo over perfect accuracy.

## Tech Stack
- Python
- Streamlit
- OpenCV
- SQLite
- Pandas
- OOP structure
- Later: YOLO for phone detection
- Later: MediaPipe for head pose estimation
- Later: ESP32/MQTT for LED/vibration alerting

## Development Rules
- Always inspect existing files before editing.
- Keep code modular.
- Do not put all logic in app.py.
- Use classes for core components.
- After code changes, run the relevant command and fix errors.
- Prefer simple explainable logic over complex AI logic.
- Do not add unnecessary dependencies without explaining why.
- Keep the project easy to present in college.

## MVP Scope
First version should include:
- Webcam or uploaded video input
- Dummy student seat mapping
- Simulated attention/distraction detection
- Dashboard with student status
- SQLite event logging
- CSV report generation

## Folder Structure
Use:
- app.py
- src/camera_manager.py
- src/student.py
- src/database_manager.py
- src/attention_analyzer.py
- src/phone_detector.py
- src/head_pose_estimator.py
- src/alert_manager.py
- src/report_generator.py
- requirements.txt
- README.md

## Testing
After changes, run:
streamlit run app.py

## Output Style
After each task, summarize:
- files changed
- what works now
- how to run
- next recommended step