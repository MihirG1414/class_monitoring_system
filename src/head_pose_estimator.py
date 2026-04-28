from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np

from src.phone_detector import DetectionBox


LOOKING_FORWARD = "LOOKING_FORWARD"
LOOKING_DOWN = "LOOKING_DOWN"
FACE_NOT_VISIBLE = "FACE_NOT_VISIBLE"


@dataclass
class HeadPoseStudentResult:
    student_id: str
    head_pose: str
    distracted_duration_seconds: float
    bounding_box: tuple[int, int, int, int] | None
    face_visible: bool


@dataclass
class HeadPoseAnalysisResult:
    student_results: dict[str, HeadPoseStudentResult]
    annotated_frame: np.ndarray | None


class MediaPipeHeadPoseEstimator:
    """Simple, explainable head-down detector based on Face Mesh landmark geometry."""

    def __init__(
        self,
        down_ratio_threshold: float = 0.58,
        down_time_threshold_seconds: float = 3.0,
        model_path: Path | None = None,
    ) -> None:
        self.down_ratio_threshold = down_ratio_threshold
        self.down_time_threshold_seconds = down_time_threshold_seconds
        self.model_path = model_path or Path("data") / "face_landmarker.task"
        self.model_url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        )

    def _ensure_model(self) -> Path:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.model_path.exists():
            urlretrieve(self.model_url, self.model_path)
        return self.model_path

    def _create_landmarker(self, running_mode_name: str):
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "MediaPipe is not installed. Run 'python -m pip install -r requirements.txt'."
            ) from error

        model_path = str(self._ensure_model())
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        return FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=getattr(RunningMode, running_mode_name),
                num_faces=6,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )

    def analyze_person_boxes(
        self,
        frame: np.ndarray,
        student_boxes: list[DetectionBox],
        sample_duration_seconds: float,
    ) -> HeadPoseAnalysisResult:
        annotated_frame = frame.copy()
        student_results: dict[str, HeadPoseStudentResult] = {}
        with self._create_landmarker("IMAGE") as landmarker:
            for index, student_box in enumerate(student_boxes, start=1):
                student_id = f"student_{index}"
                crop = self._crop_student(frame, student_box)
                if crop.size == 0:
                    student_results[student_id] = HeadPoseStudentResult(
                        student_id=student_id,
                        head_pose=FACE_NOT_VISIBLE,
                        distracted_duration_seconds=sample_duration_seconds,
                        bounding_box=None,
                        face_visible=False,
                    )
                    continue
                result = landmarker.detect(self._to_mp_image(crop))
                face_landmarks_list = getattr(result, "face_landmarks", None)
                if not face_landmarks_list:
                    self._draw_student_box(annotated_frame, student_box, FACE_NOT_VISIBLE)
                    student_results[student_id] = HeadPoseStudentResult(
                        student_id=student_id,
                        head_pose=FACE_NOT_VISIBLE,
                        distracted_duration_seconds=sample_duration_seconds,
                        bounding_box=(int(student_box.x1), int(student_box.y1), int(student_box.x2), int(student_box.y2)),
                        face_visible=False,
                    )
                    continue

                landmarks = face_landmarks_list[0]
                head_pose = self._classify_head_pose(landmarks)
                distracted_duration_seconds = sample_duration_seconds if head_pose == LOOKING_DOWN else 0.0
                self._draw_student_box(annotated_frame, student_box, head_pose)
                student_results[student_id] = HeadPoseStudentResult(
                    student_id=student_id,
                    head_pose=head_pose,
                    distracted_duration_seconds=distracted_duration_seconds,
                    bounding_box=(int(student_box.x1), int(student_box.y1), int(student_box.x2), int(student_box.y2)),
                    face_visible=True,
                )
        return HeadPoseAnalysisResult(student_results=student_results, annotated_frame=annotated_frame)

    def analyze_file(self, file_path: Path, seat_map) -> HeadPoseAnalysisResult:
        extension = file_path.suffix.lower()
        if extension in {".jpg", ".jpeg", ".png"}:
            frame = cv2.imread(str(file_path))
            return self.analyze_person_boxes(frame, seat_map, sample_duration_seconds=1.0)

        if extension in {".mp4", ".avi", ".mov", ".mkv"}:
            frame = cv2.VideoCapture(str(file_path))
            frame.release()
            raise RuntimeError("Use classroom analysis for video-based head-pose estimation.")

        raise RuntimeError(f"Unsupported file type for head-pose estimation: {file_path.suffix}")

    def _to_mp_image(self, frame: np.ndarray):
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    def _classify_head_pose(self, landmarks) -> str:
        left_eye_y = landmarks[33].y
        right_eye_y = landmarks[263].y
        eye_mid_y = (left_eye_y + right_eye_y) / 2
        nose_y = landmarks[1].y
        chin_y = landmarks[152].y
        denominator = max(chin_y - eye_mid_y, 1e-6)
        nose_drop_ratio = (nose_y - eye_mid_y) / denominator
        if nose_drop_ratio > self.down_ratio_threshold:
            return LOOKING_DOWN
        return LOOKING_FORWARD

    def _crop_student(self, frame: np.ndarray, student_box: DetectionBox) -> np.ndarray:
        x1 = max(0, int(student_box.x1))
        y1 = max(0, int(student_box.y1))
        x2 = min(frame.shape[1], int(student_box.x2))
        y2 = min(frame.shape[0], int(student_box.y2))
        return frame[y1:y2, x1:x2]

    def _draw_student_box(self, frame: np.ndarray, student_box: DetectionBox, head_pose: str) -> None:
        x1, y1, x2, y2 = int(student_box.x1), int(student_box.y1), int(student_box.x2), int(student_box.y2)
        color = (80, 220, 120) if head_pose == LOOKING_FORWARD else (80, 120, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            head_pose,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
