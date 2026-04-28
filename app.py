from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.alert_manager import AlertManager
from src.head_pose_estimator import MediaPipeHeadPoseEstimator
from src.models import StudentSeat
from src.monitoring import ClassroomMonitoringService
from src.phone_detector import PhoneDetectionOrchestrator, YoloPhoneDetector
from src.reporting import CSVReportGenerator
from src.simulator import DummyAttentionDetector
from src.storage import SQLiteEventRepository


st.set_page_config(
    page_title="Intelligent Classroom Attention Monitoring System",
    layout="wide",
)


TEACHER_USERNAME = "teacher"
TEACHER_PASSWORD = "classroom123"
DEFAULT_DETECTION_MODE = "Simulated"
CLASSROOM_ANALYSIS_MODE = "Classroom Full Analysis"


def read_uploaded_frame(uploaded_file) -> tuple[np.ndarray | None, str]:
    file_extension = Path(uploaded_file.name).suffix.lower()
    file_bytes = uploaded_file.getvalue()
    if file_extension in {".jpg", ".jpeg", ".png"}:
        image_array = np.frombuffer(file_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        return frame, uploaded_file.name
    if file_extension in {".mp4", ".avi", ".mov", ".mkv"}:
        temp_path = Path("data") / f"temp_{uploaded_file.name}"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_bytes(file_bytes)
        capture = cv2.VideoCapture(str(temp_path))
        success, frame = capture.read()
        capture.release()
        temp_path.unlink(missing_ok=True)
        if not success:
            return None, uploaded_file.name
        return frame, uploaded_file.name
    return None, uploaded_file.name


def save_uploaded_file(uploaded_file) -> Path:
    temp_path = Path("data") / f"temp_{uploaded_file.name}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(uploaded_file.getvalue())
    return temp_path


def to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def bootstrap_service() -> ClassroomMonitoringService:
    if "monitoring_service" not in st.session_state:
        database_path = Path("data") / "classroom_events.db"
        repository = SQLiteEventRepository(database_path)
        detector = DummyAttentionDetector()
        phone_detector = PhoneDetectionOrchestrator(YoloPhoneDetector())
        head_pose_estimator = MediaPipeHeadPoseEstimator()
        alert_manager = AlertManager()
        seat_map = [
            StudentSeat("student_1", "student_1", "STU001", 0, 0.00, 0.00, 0.33, 0.50),
            StudentSeat("student_2", "student_2", "STU002", 0, 0.33, 0.00, 0.66, 0.50),
            StudentSeat("student_3", "student_3", "STU003", 0, 0.66, 0.00, 1.00, 0.50),
            StudentSeat("student_4", "student_4", "STU004", 0, 0.00, 0.50, 0.33, 1.00),
            StudentSeat("student_5", "student_5", "STU005", 0, 0.33, 0.50, 0.66, 1.00),
            StudentSeat("student_6", "student_6", "STU006", 0, 0.66, 0.50, 1.00, 1.00),
        ]
        st.session_state.monitoring_service = ClassroomMonitoringService(
            detector=detector,
            repository=repository,
            seat_map=seat_map,
            phone_detector=phone_detector,
            head_pose_estimator=head_pose_estimator,
            alert_manager=alert_manager,
        )
    return st.session_state.monitoring_service


def render_summary_cards(summary: dict) -> None:
    columns = st.columns(5)
    columns[0].metric("Total Students", summary["total_students"])
    columns[1].metric("Attentive", summary["attentive_students"])
    columns[2].metric("Distracted", summary["distracted_students"])
    columns[3].metric("Average Attention", f'{summary["average_attention"]:.1f}%')
    columns[4].metric("Phone Detections", summary["total_phone_detections"])


def initialize_auth_state() -> None:
    if "teacher_authenticated" not in st.session_state:
        st.session_state.teacher_authenticated = False


def render_login_screen() -> None:
    st.title("Teacher Login")
    st.caption("Sign in to access the classroom attention dashboard.")
    with st.form("teacher_login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True, type="primary")
    if submitted:
        if username == TEACHER_USERNAME and password == TEACHER_PASSWORD:
            st.session_state.teacher_authenticated = True
            st.rerun()
        else:
            st.error("Invalid username or password.")


def main() -> None:
    initialize_auth_state()
    if not st.session_state.teacher_authenticated:
        render_login_screen()
        return

    st.title("Intelligent Classroom Attention Monitoring System")
    st.caption("Upload a classroom image or video to detect students, phones, face visibility, and attention with downloadable reporting.")

    service = bootstrap_service()
    report_generator = CSVReportGenerator()

    with st.sidebar:
        st.success("Logged in as teacher")
        if st.button("Logout", use_container_width=True):
            st.session_state.teacher_authenticated = False
            st.session_state.pop("monitoring_service", None)
            st.session_state.pop("last_uploaded_frame", None)
            st.session_state.pop("last_annotated_frame", None)
            st.session_state.pop("report_path", None)
            st.rerun()

        st.header("Controls")
        detection_mode = st.radio(
            "Detection mode",
            options=[CLASSROOM_ANALYSIS_MODE, DEFAULT_DETECTION_MODE],
            index=0,
        )
        if detection_mode == CLASSROOM_ANALYSIS_MODE:
            uploaded_file = st.file_uploader(
                "Upload classroom image or video",
                type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"],
                key="classroom_upload",
            )
            if st.button("Run Classroom Analysis", type="primary", use_container_width=True):
                if uploaded_file is None:
                    st.error("Upload a classroom image or video first.")
                else:
                    temp_path = save_uploaded_file(uploaded_file)
                    try:
                        analysis = service.run_classroom_analysis(
                            file_path=temp_path,
                            source_name=uploaded_file.name,
                        )
                    except RuntimeError as error:
                        st.error(str(error))
                    else:
                        frame, _ = read_uploaded_frame(uploaded_file)
                        if frame is not None:
                            st.session_state.last_uploaded_frame = to_rgb(frame)
                        if getattr(analysis, "annotated_frame", None) is not None:
                            st.session_state.last_annotated_frame = to_rgb(analysis.annotated_frame)
                        st.success(f"Processed {uploaded_file.name}. Students were labeled automatically as student_1, student_2, and so on.")
                    finally:
                        temp_path.unlink(missing_ok=True)
        else:
            cycles = st.slider("Simulation cycles", min_value=1, max_value=10, value=1)
            if st.button("Run Simulation", type="primary", use_container_width=True):
                service.run_monitoring_cycle(cycles=cycles)
                st.success(f"Recorded {cycles} monitoring cycle(s).")
        if st.button("Reset Session Data", use_container_width=True):
            service.reset_monitoring_data()
            st.session_state.pop("last_uploaded_frame", None)
            st.session_state.pop("last_annotated_frame", None)
            st.warning("Database events cleared and metrics reset.")

        st.header("Reports")
        if st.button("Generate CSV Report", use_container_width=True):
            report_path = report_generator.generate(service.get_session_report_dataframe())
            st.session_state.report_path = report_path
            st.success(f"Report created: {report_path.name}")

        report_path = st.session_state.get("report_path")
        if report_path:
            with open(report_path, "rb") as csv_file:
                st.download_button(
                    label="Download Latest CSV",
                    data=csv_file.read(),
                    file_name=report_path.name,
                    mime="text/csv",
                    use_container_width=True,
                )

    summary = service.get_summary_metrics()
    session_report_dataframe = service.get_session_report_dataframe()
    render_summary_cards(summary)

    if detection_mode == CLASSROOM_ANALYSIS_MODE:
        st.subheader("Classroom Analysis Preview")
        preview_columns = st.columns(2)
        original_frame = st.session_state.get("last_uploaded_frame")
        annotated_frame = st.session_state.get("last_annotated_frame")
        if original_frame is not None:
            preview_columns[0].image(original_frame, caption="Input Frame", use_container_width=True)
        else:
            preview_columns[0].info("Upload a classroom image or video to see the original frame.")
        if annotated_frame is not None:
            preview_columns[1].image(annotated_frame, caption="Detected Students, Phones, and Attention Labels", use_container_width=True)
        else:
            preview_columns[1].info("Processed detection overlays will appear here after you run the selected mode.")

    st.subheader("Student Attention Dashboard")
    dashboard_dataframe = service.get_dashboard_dataframe()
    if not dashboard_dataframe.empty:
        st.dataframe(dashboard_dataframe, use_container_width=True, hide_index=True)
    else:
        st.write("Run classroom analysis to populate the student table.")

    st.subheader("Simulated Non-Disruptive Alerts")
    if not dashboard_dataframe.empty:
        alert_preview = dashboard_dataframe[["Seat", "Student Name", "Status", "Latest Alert"]]
        st.dataframe(alert_preview, use_container_width=True, hide_index=True)
    else:
        st.write("Alert status will appear here after analysis.")

    st.subheader("Attention Analytics")
    if not session_report_dataframe.empty:
        attention_chart_data = (
            session_report_dataframe[["name", "attention_percentage"]]
            .rename(columns={"name": "Student", "attention_percentage": "Attention Percentage"})
            .set_index("Student")
        )
        distracted_chart_data = (
            session_report_dataframe[["name", "distracted_duration"]]
            .rename(columns={"name": "Student", "distracted_duration": "Distracted Duration (s)"})
            .set_index("Student")
        )
        chart_columns = st.columns(2)
        chart_columns[0].caption("Attention Percentage by Student")
        chart_columns[0].bar_chart(attention_chart_data, use_container_width=True)
        chart_columns[1].caption("Distracted Duration by Student")
        chart_columns[1].bar_chart(distracted_chart_data, use_container_width=True)
    else:
        st.write("Run monitoring first to generate attention analytics.")

    st.subheader("Session Report Preview")
    st.dataframe(session_report_dataframe, use_container_width=True, hide_index=True)

    st.subheader("Student Labels")
    if service.seat_map:
        label_columns = st.columns(3)
        for index, student in enumerate(service.seat_map):
            label_columns[index % 3].info(
                f"ID: {student.seat_id}\n\n"
                f"Name: {student.student_name}\n\n"
                f"Roll No: {student.roll_number}"
            )
    else:
        st.write("No students detected yet in the current classroom analysis.")

    st.subheader("Recent Stored Events")
    recent_events = service.get_recent_events(limit=15)
    if not recent_events.empty:
        st.dataframe(recent_events, use_container_width=True, hide_index=True)
    else:
        st.write("No events recorded yet. Run a simulation cycle to populate the dashboard.")


if __name__ == "__main__":
    main()
