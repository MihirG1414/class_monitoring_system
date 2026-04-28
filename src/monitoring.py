from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd

from src.alert_manager import AlertManager
from src.head_pose_estimator import (
    FACE_NOT_VISIBLE,
    LOOKING_DOWN,
    LOOKING_FORWARD,
    MediaPipeHeadPoseEstimator,
)
from src.models import AttentionEvent, StudentSeat
from src.phone_detector import DetectionBox, PhoneDetectionOrchestrator
from src.reporting import CSVReportGenerator
from src.simulator import DummyAttentionDetector
from src.storage import SQLiteEventRepository


class ClassroomMonitoringService:
    def __init__(
        self,
        detector: DummyAttentionDetector,
        repository: SQLiteEventRepository,
        seat_map: list[StudentSeat],
        phone_detector: PhoneDetectionOrchestrator | None = None,
        head_pose_estimator: MediaPipeHeadPoseEstimator | None = None,
        alert_manager: AlertManager | None = None,
    ) -> None:
        self.detector = detector
        self.repository = repository
        self.seat_map = seat_map
        self.phone_detector = phone_detector
        self.head_pose_estimator = head_pose_estimator
        self.alert_manager = alert_manager or AlertManager()
        self._latest_dashboard = self._build_default_dashboard()
        self._alert_log = {
            student.seat_id: []
            for student in self.seat_map
        }
        self._sync_phone_counts()

    def _build_default_dashboard(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "seat_id": student.seat_id,
                    "student_name": student.student_name,
                    "roll_number": student.roll_number,
                    "status": "Not Monitored",
                    "attention_percentage": 0.0,
                    "distracted_duration_seconds": 0,
                    "alert_count": student.initial_alert_count,
                    "head_pose": "SIMULATED",
                    "phone_detection_count": 0,
                    "phone_visible": "No",
                    "face_visible": "Unknown",
                    "latest_alert": "No alert triggered",
                }
                for student in self.seat_map
            ]
        )

    def _get_previous_alerts(self) -> dict[str, int]:
        dashboard = self._latest_dashboard.set_index("seat_id")
        return dashboard["alert_count"].to_dict()

    def _sync_phone_counts(self) -> None:
        if self._latest_dashboard.empty or "seat_id" not in self._latest_dashboard.columns:
            return
        counts = self.repository.fetch_phone_detection_counts()
        self._latest_dashboard["phone_detection_count"] = self._latest_dashboard["seat_id"].map(counts).fillna(0).astype(int)

    def _sync_alert_labels(self) -> None:
        if self._latest_dashboard.empty or "seat_id" not in self._latest_dashboard.columns:
            return
        self._latest_dashboard["latest_alert"] = self._latest_dashboard["seat_id"].map(
            lambda seat_id: " | ".join(self._alert_log.get(seat_id, [])[-4:]) if self._alert_log.get(seat_id) else "No alert triggered"
        )

    def _trigger_simulated_alerts_for_status(self, seat_id: str, status: str) -> None:
        if status not in {"Distracted", "Not Attentive"}:
            return
        self._alert_log.setdefault(seat_id, []).extend(self.alert_manager.trigger_all(seat_id))

    def run_monitoring_cycle(self, cycles: int = 1) -> None:
        for _ in range(cycles):
            previous_phone_counts = self._latest_dashboard.set_index("seat_id")["phone_detection_count"].to_dict()
            events = self.detector.detect(self.seat_map, self._get_previous_alerts())
            for event in events:
                self._trigger_simulated_alerts_for_status(event.seat_id, event.status)
            self.repository.save_events(events)
            self._latest_dashboard = pd.DataFrame(
                [
                    {
                        "seat_id": event.seat_id,
                        "student_name": event.student_name,
                        "roll_number": event.roll_number,
                        "status": event.status,
                        "attention_percentage": event.attention_percentage,
                        "distracted_duration_seconds": event.distracted_duration_seconds,
                        "alert_count": event.alert_count,
                        "head_pose": event.head_pose,
                        "phone_detection_count": int(previous_phone_counts.get(event.seat_id, 0)),
                        "phone_visible": "No",
                        "face_visible": "Yes",
                        "latest_alert": "No alert triggered",
                    }
                    for event in events
                ]
            )
            self._sync_phone_counts()
            self._sync_alert_labels()

    def _set_dynamic_roster(self, person_boxes: list[DetectionBox]) -> None:
        if not person_boxes:
            self.seat_map = []
            self._latest_dashboard = pd.DataFrame()
            return
        self.seat_map = [
            StudentSeat(
                seat_id=f"student_{index}",
                student_name=f"student_{index}",
                roll_number=f"STU{index:03d}",
            )
            for index in range(1, len(person_boxes) + 1)
        ]
        self._alert_log = {student.seat_id: self._alert_log.get(student.seat_id, []) for student in self.seat_map}

    def run_classroom_analysis(self, file_path: Path, source_name: str) -> object:
        if self.phone_detector is None or self.head_pose_estimator is None:
            raise RuntimeError("Classroom analysis requires both YOLO and MediaPipe components.")

        extension = file_path.suffix.lower()
        if extension in {".jpg", ".jpeg", ".png"}:
            frame = cv2.imread(str(file_path))
            if frame is None:
                raise RuntimeError("Could not read the uploaded classroom image.")
            return self._analyze_image_frame(frame, source_name)
        if extension in {".mp4", ".avi", ".mov", ".mkv"}:
            return self._analyze_video_file(file_path, source_name)
        raise RuntimeError(f"Unsupported classroom analysis file type: {file_path.suffix}")

    def _analyze_image_frame(self, frame, source_name: str) -> object:
        classroom_detections = self.phone_detector.detect_classroom(frame)
        if not classroom_detections.person_boxes:
            self.seat_map = []
            self._latest_dashboard = pd.DataFrame(
                columns=[
                    "seat_id",
                    "student_name",
                    "roll_number",
                    "status",
                    "attention_percentage",
                    "distracted_duration_seconds",
                    "alert_count",
                    "head_pose",
                    "phone_detection_count",
                    "phone_visible",
                    "face_visible",
                    "latest_alert",
                ]
            )
            return classroom_detections

        self._set_dynamic_roster(classroom_detections.person_boxes)
        head_pose_analysis = self.head_pose_estimator.analyze_person_boxes(
            frame=frame,
            student_boxes=classroom_detections.person_boxes,
            sample_duration_seconds=1.0,
        )
        phone_assignments = self.phone_detector.assign_phones_to_people(
            classroom_detections.person_boxes,
            classroom_detections.phone_boxes,
        )
        student_rows = self._build_student_rows(
            student_boxes=classroom_detections.person_boxes,
            head_pose_results=head_pose_analysis.student_results,
            phone_assignments=phone_assignments,
            source_name=source_name,
            sample_duration_seconds=1.0,
            previous_alerts=self._get_previous_alerts() if not self._latest_dashboard.empty else {},
        )
        self._persist_classroom_results(student_rows, source_name)
        head_pose_analysis.annotated_frame = self._annotate_classroom_frame(
            frame=classroom_detections.annotated_frame,
            student_rows=student_rows,
        )
        return head_pose_analysis

    def _analyze_video_file(self, file_path: Path, source_name: str) -> object:
        capture = cv2.VideoCapture(str(file_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        sample_interval_seconds = 1.0
        frame_interval = max(1, int(round(fps * sample_interval_seconds)))
        frame_index = 0
        anchor_boxes: list[DetectionBox] = []
        aggregate_stats: dict[str, dict] = {}
        latest_frame = None
        latest_student_rows: list[dict] = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            classroom_detections = self.phone_detector.detect_classroom(frame)
            if not anchor_boxes and classroom_detections.person_boxes:
                anchor_boxes = classroom_detections.person_boxes
                self._set_dynamic_roster(anchor_boxes)
                aggregate_stats = {
                    student.seat_id: {
                        "samples": 0,
                        "attentive_samples": 0,
                        "distracted_samples": 0,
                        "phone_detection_count": 0,
                        "alert_count": 0,
                        "face_visible": False,
                        "phone_visible": False,
                        "head_pose": FACE_NOT_VISIBLE,
                    }
                    for student in self.seat_map
                }
            if not anchor_boxes:
                frame_index += 1
                continue

            matched_boxes = self._match_person_boxes(anchor_boxes, classroom_detections.person_boxes)
            present_boxes = [box for box in matched_boxes if box is not None]
            head_pose_results = {}
            if present_boxes:
                head_pose_analysis = self.head_pose_estimator.analyze_person_boxes(
                    frame=frame,
                    student_boxes=present_boxes,
                    sample_duration_seconds=sample_interval_seconds,
                )
                head_pose_results = head_pose_analysis.student_results
                latest_frame = head_pose_analysis.annotated_frame
            latest_student_rows = self._build_student_rows_for_video(
                matched_boxes=matched_boxes,
                head_pose_results=head_pose_results,
                phone_boxes=classroom_detections.phone_boxes,
                source_name=source_name,
                aggregate_stats=aggregate_stats,
                sample_duration_seconds=sample_interval_seconds,
            )
            frame_index += 1

        capture.release()
        if not latest_student_rows:
            self.seat_map = []
            self._latest_dashboard = pd.DataFrame()
            return type("EmptyAnalysis", (), {"annotated_frame": None, "student_results": {}})()

        self._persist_classroom_results(latest_student_rows, source_name)
        annotated_frame = self._annotate_classroom_frame(
            frame=latest_frame if latest_frame is not None else cv2.imread(str(file_path)),
            student_rows=latest_student_rows,
        )
        return type("VideoAnalysis", (), {"annotated_frame": annotated_frame, "student_results": {}})()

    def _build_student_rows(
        self,
        student_boxes: list[DetectionBox],
        head_pose_results: dict,
        phone_assignments: list[list[DetectionBox]],
        source_name: str,
        sample_duration_seconds: float,
        previous_alerts: dict[str, int],
    ) -> list[dict]:
        rows: list[dict] = []
        for index, student_box in enumerate(student_boxes, start=1):
            student = self.seat_map[index - 1]
            head_result = head_pose_results.get(student.seat_id)
            phone_boxes = phone_assignments[index - 1]
            phone_visible = len(phone_boxes) > 0
            face_visible = bool(head_result and head_result.face_visible)
            head_pose = head_result.head_pose if head_result else FACE_NOT_VISIBLE
            is_attentive = head_pose == LOOKING_FORWARD and face_visible and not phone_visible
            status = "Attentive" if is_attentive else "Not Attentive"
            attention_percentage = 100.0 if is_attentive else 0.0
            distracted_duration_seconds = 0 if is_attentive else int(sample_duration_seconds)
            alert_count = int(previous_alerts.get(student.seat_id, 0))
            if not is_attentive:
                alert_count += 1
                self._trigger_simulated_alerts_for_status(student.seat_id, "Distracted")
            rows.append(
                {
                    "seat_id": student.seat_id,
                    "student_name": student.student_name,
                    "roll_number": student.roll_number,
                    "status": status,
                    "attention_percentage": attention_percentage,
                    "distracted_duration_seconds": distracted_duration_seconds,
                    "alert_count": alert_count,
                    "head_pose": head_pose,
                    "phone_detection_count": len(phone_boxes),
                    "phone_visible": "Yes" if phone_visible else "No",
                    "face_visible": "Yes" if face_visible else "No",
                    "latest_alert": "No alert triggered",
                    "phone_boxes": phone_boxes,
                    "source_name": source_name,
                }
            )
        return rows

    def _build_student_rows_for_video(
        self,
        matched_boxes: list[DetectionBox | None],
        head_pose_results: dict,
        phone_boxes: list[DetectionBox],
        source_name: str,
        aggregate_stats: dict[str, dict],
        sample_duration_seconds: float,
    ) -> list[dict]:
        rows: list[dict] = []
        present_boxes = [box for box in matched_boxes if box is not None]
        present_phone_assignments = self.phone_detector.assign_phones_to_people(present_boxes, phone_boxes) if present_boxes else []
        assignment_index = 0
        for index, matched_box in enumerate(matched_boxes, start=1):
            student = self.seat_map[index - 1]
            stats = aggregate_stats[student.seat_id]
            stats["samples"] += 1
            if matched_box is None:
                head_pose = FACE_NOT_VISIBLE
                face_visible = False
                phone_visible = False
                assigned_phone_boxes = []
            else:
                head_result = head_pose_results.get(f"student_{assignment_index + 1}")
                assigned_phone_boxes = present_phone_assignments[assignment_index] if assignment_index < len(present_phone_assignments) else []
                assignment_index += 1
                head_pose = head_result.head_pose if head_result else FACE_NOT_VISIBLE
                face_visible = bool(head_result and head_result.face_visible)
                phone_visible = len(assigned_phone_boxes) > 0
            is_attentive = head_pose == LOOKING_FORWARD and face_visible and not phone_visible
            if is_attentive:
                stats["attentive_samples"] += 1
            else:
                stats["distracted_samples"] += 1
                stats["alert_count"] += 1
                self._trigger_simulated_alerts_for_status(student.seat_id, "Distracted")
            stats["phone_detection_count"] += len(assigned_phone_boxes)
            stats["face_visible"] = face_visible
            stats["phone_visible"] = phone_visible or stats["phone_visible"]
            stats["head_pose"] = head_pose

            total_session_duration = stats["samples"] * sample_duration_seconds
            distracted_duration_seconds = stats["distracted_samples"] * sample_duration_seconds
            attention_percentage = round(
                (stats["attentive_samples"] / stats["samples"]) * 100 if stats["samples"] else 0.0,
                2,
            )
            rows.append(
                {
                    "seat_id": student.seat_id,
                    "student_name": student.student_name,
                    "roll_number": student.roll_number,
                    "status": "Attentive" if is_attentive else "Not Attentive",
                    "attention_percentage": attention_percentage,
                    "distracted_duration_seconds": int(distracted_duration_seconds),
                    "alert_count": int(stats["alert_count"]),
                    "head_pose": head_pose,
                    "phone_detection_count": int(stats["phone_detection_count"]),
                    "phone_visible": "Yes" if stats["phone_visible"] else "No",
                    "face_visible": "Yes" if face_visible else "No",
                    "latest_alert": "No alert triggered",
                    "phone_boxes": assigned_phone_boxes,
                    "source_name": source_name,
                    "total_session_duration": total_session_duration,
                }
            )
        return rows

    def _match_person_boxes(
        self,
        anchor_boxes: list[DetectionBox],
        current_boxes: list[DetectionBox],
    ) -> list[DetectionBox | None]:
        remaining_boxes = current_boxes.copy()
        matched_boxes: list[DetectionBox | None] = []
        for anchor_box in anchor_boxes:
            if not remaining_boxes:
                matched_boxes.append(None)
                continue
            best_box = min(remaining_boxes, key=lambda box: self._box_distance(anchor_box, box))
            matched_boxes.append(best_box)
            remaining_boxes.remove(best_box)
        return matched_boxes

    def _box_distance(self, first_box: DetectionBox, second_box: DetectionBox) -> float:
        first_x, first_y = first_box.center
        second_x, second_y = second_box.center
        return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5

    def _persist_classroom_results(self, student_rows: list[dict], source_name: str) -> None:
        self._latest_dashboard = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"phone_boxes", "source_name", "total_session_duration"}
                }
                for row in student_rows
            ]
        )
        attention_events = [
            AttentionEvent(
                timestamp=datetime.now(),
                seat_id=row["seat_id"],
                student_name=row["student_name"],
                roll_number=row["roll_number"],
                status=row["status"],
                attention_percentage=row["attention_percentage"],
                distracted_duration_seconds=row["distracted_duration_seconds"],
                alert_count=row["alert_count"],
                head_pose=row["head_pose"],
                source_name=source_name,
            )
            for row in student_rows
        ]
        phone_events = self.phone_detector.create_phone_events(source_name=source_name, student_rows=student_rows)
        self.repository.save_events(attention_events)
        self.repository.save_phone_events(phone_events)
        self._sync_alert_labels()

    def _annotate_classroom_frame(self, frame, student_rows: list[dict]):
        if frame is None:
            return None
        return frame.copy()

    def run_head_pose_detection(self, file_path, source_name: str) -> object:
        raise RuntimeError("Use Classroom Full Analysis mode for head-pose classroom results.")

    def run_phone_detection(self, frame, source_name: str) -> tuple[int, object]:
        if self.phone_detector is None:
            raise RuntimeError("YOLO phone detector is not configured.")
        phone_events, annotated_frame = self.phone_detector.detect_and_assign(
            frame=frame,
            seat_map=self.seat_map,
            source_name=source_name,
        )
        self.repository.save_phone_events(phone_events)
        self._sync_phone_counts()
        return len(phone_events), annotated_frame

    def get_dashboard_dataframe(self) -> pd.DataFrame:
        dataframe = self._latest_dashboard.copy()
        dataframe = dataframe.rename(
            columns={
                "seat_id": "Seat",
                "student_name": "Student Name",
                "roll_number": "Roll Number",
                "status": "Status",
                "attention_percentage": "Attention Percentage",
                "distracted_duration_seconds": "Distracted Duration (s)",
                "alert_count": "Alert Count",
                "head_pose": "Head Pose",
                "phone_detection_count": "Phone Detection Count",
                "phone_visible": "Phone Visible",
                "face_visible": "Face Visible",
                "latest_alert": "Latest Alert",
            }
        )
        return dataframe

    def get_dashboard_records(self) -> list[dict]:
        return self.get_dashboard_dataframe().to_dict(orient="records")

    def get_session_report_dataframe(self) -> pd.DataFrame:
        report_generator = CSVReportGenerator()
        return report_generator.build_session_report(
            attention_events=self.repository.fetch_events(),
            phone_events=self.repository.fetch_phone_events(),
            seat_map=self.seat_map,
        )

    def get_summary_metrics(self) -> dict:
        dashboard = self._latest_dashboard
        if dashboard.empty or "status" not in dashboard.columns:
            return {
                "total_students": 0,
                "attentive_students": 0,
                "distracted_students": 0,
                "average_attention": 0.0,
                "total_phone_detections": 0,
            }
        total_students = len(dashboard.index)
        attentive_students = int((dashboard["status"] == "Attentive").sum())
        distracted_students = int((dashboard["status"] != "Attentive").sum())
        average_attention = float(dashboard["attention_percentage"].mean()) if total_students else 0.0
        total_phone_detections = int(dashboard["phone_detection_count"].sum()) if total_students else 0
        return {
            "total_students": total_students,
            "attentive_students": attentive_students,
            "distracted_students": distracted_students,
            "average_attention": average_attention,
            "total_phone_detections": total_phone_detections,
        }

    def get_recent_events(self, limit: int = 15) -> pd.DataFrame:
        return self.repository.fetch_recent_events(limit=limit)

    def reset_monitoring_data(self) -> None:
        self.repository.clear()
        self._latest_dashboard = self._build_default_dashboard()
        self._alert_log = {
            student.seat_id: []
            for student in self.seat_map
        }
