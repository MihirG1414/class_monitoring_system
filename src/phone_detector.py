from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from src.models import PhoneDetectionEvent, StudentSeat


@dataclass
class DetectionBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)


@dataclass
class ClassroomDetections:
    person_boxes: list[DetectionBox]
    phone_boxes: list[DetectionBox]
    annotated_frame: np.ndarray


class YoloPhoneDetector:
    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        person_confidence_threshold: float = 0.18,
        phone_confidence_threshold: float = 0.28,
        image_size: int = 1280,
    ) -> None:
        self.model_path = model_path
        self.person_confidence_threshold = person_confidence_threshold
        self.phone_confidence_threshold = phone_confidence_threshold
        self.image_size = image_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as error:
                raise RuntimeError(
                    "Ultralytics is not installed. Run 'python -m pip install -r requirements.txt'."
                ) from error
            self._model = YOLO(self.model_path)
        return self._model

    def detect_classroom_objects(self, frame: np.ndarray) -> ClassroomDetections:
        model = self._load_model()
        results = model.predict(
            source=frame,
            conf=min(self.person_confidence_threshold, self.phone_confidence_threshold),
            imgsz=self.image_size,
            iou=0.45,
            max_det=300,
            verbose=False,
        )
        person_boxes: list[DetectionBox] = []
        phone_boxes: list[DetectionBox] = []
        annotated_frame = frame.copy()
        for result in results:
            names = result.names
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = str(names[class_id]).lower()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0].item())
                detection_box = DetectionBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=confidence,
                    class_name=class_name,
                )
                if class_name == "person" and confidence >= self.person_confidence_threshold:
                    person_boxes.append(detection_box)
                elif class_name == "cell phone" and confidence >= self.phone_confidence_threshold:
                    phone_boxes.append(detection_box)

        person_boxes = self._sort_boxes(person_boxes)
        phone_boxes = self._filter_phone_boxes(phone_boxes, person_boxes)
        phone_boxes = self._refine_phone_boxes_with_person_crops(frame, person_boxes, phone_boxes)
        self._draw_detections(annotated_frame, person_boxes, phone_boxes)
        return ClassroomDetections(
            person_boxes=person_boxes,
            phone_boxes=phone_boxes,
            annotated_frame=annotated_frame,
        )

    def _sort_boxes(self, boxes: list[DetectionBox]) -> list[DetectionBox]:
        return sorted(boxes, key=lambda box: (box.y1, box.x1))

    def _filter_phone_boxes(
        self,
        phone_boxes: list[DetectionBox],
        person_boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        filtered_boxes: list[DetectionBox] = []
        for phone_box in phone_boxes:
            candidate_person = self._best_matching_person(phone_box, person_boxes)
            if candidate_person is None:
                continue
            if not self._looks_like_real_phone(phone_box, candidate_person):
                continue
            filtered_boxes.append(phone_box)
        return filtered_boxes

    def _refine_phone_boxes_with_person_crops(
        self,
        frame: np.ndarray,
        person_boxes: list[DetectionBox],
        phone_boxes: list[DetectionBox],
    ) -> list[DetectionBox]:
        refined_boxes = list(phone_boxes)
        model = self._load_model()
        for person_box in person_boxes:
            crop = self._crop_box(frame, person_box)
            if crop.size == 0:
                continue
            results = model.predict(
                source=crop,
                conf=max(0.16, self.phone_confidence_threshold - 0.08),
                imgsz=960,
                iou=0.4,
                max_det=20,
                verbose=False,
            )
            for result in results:
                names = result.names
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    class_name = str(names[class_id]).lower()
                    if class_name != "cell phone":
                        continue
                    confidence = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    translated_box = DetectionBox(
                        x1=person_box.x1 + float(x1),
                        y1=person_box.y1 + float(y1),
                        x2=person_box.x1 + float(x2),
                        y2=person_box.y1 + float(y2),
                        confidence=confidence,
                        class_name=class_name,
                    )
                    if not self._looks_like_real_phone(translated_box, person_box):
                        continue
                    refined_boxes.append(translated_box)
        return self._deduplicate_phone_boxes(refined_boxes)

    def _deduplicate_phone_boxes(self, phone_boxes: list[DetectionBox]) -> list[DetectionBox]:
        deduplicated: list[DetectionBox] = []
        for phone_box in sorted(phone_boxes, key=lambda box: box.confidence, reverse=True):
            if any(self._intersection_over_union(phone_box, kept_box) > 0.45 for kept_box in deduplicated):
                continue
            deduplicated.append(phone_box)
        return deduplicated

    def _intersection_over_union(self, first_box: DetectionBox, second_box: DetectionBox) -> float:
        x_left = max(first_box.x1, second_box.x1)
        y_top = max(first_box.y1, second_box.y1)
        x_right = min(first_box.x2, second_box.x2)
        y_bottom = min(first_box.y2, second_box.y2)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        intersection = (x_right - x_left) * (y_bottom - y_top)
        first_area = first_box.width * first_box.height
        second_area = second_box.width * second_box.height
        union = max(first_area + second_area - intersection, 1.0)
        return intersection / union

    def _crop_box(self, frame: np.ndarray, box: DetectionBox) -> np.ndarray:
        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        x2 = min(frame.shape[1], int(box.x2))
        y2 = min(frame.shape[0], int(box.y2))
        return frame[y1:y2, x1:x2]

    def _best_matching_person(
        self,
        phone_box: DetectionBox,
        person_boxes: list[DetectionBox],
    ) -> DetectionBox | None:
        best_person = None
        best_score = None
        for person_box in person_boxes:
            score = self._center_distance(phone_box, person_box)
            if best_score is None or score < best_score:
                best_score = score
                best_person = person_box
        return best_person

    def _center_distance(self, first_box: DetectionBox, second_box: DetectionBox) -> float:
        first_x, first_y = first_box.center
        second_x, second_y = second_box.center
        return ((first_x - second_x) ** 2 + (first_y - second_y) ** 2) ** 0.5

    def _looks_like_real_phone(
        self,
        phone_box: DetectionBox,
        person_box: DetectionBox,
    ) -> bool:
        phone_area = phone_box.width * phone_box.height
        person_area = max(person_box.width * person_box.height, 1.0)
        area_ratio = phone_area / person_area
        aspect_ratio = phone_box.width / max(phone_box.height, 1.0)
        center_x, center_y = phone_box.center

        # Filter obvious false positives such as ties, faces, or large torso patches.
        if area_ratio < 0.001 or area_ratio > 0.12:
            return False
        if aspect_ratio < 0.22 or aspect_ratio > 2.0:
            return False
        if center_x < person_box.x1 - (person_box.width * 0.1) or center_x > person_box.x2 + (person_box.width * 0.1):
            return False
        if center_y < person_box.y1 + (person_box.height * 0.12):
            return False
        if center_y > person_box.y2 + (person_box.height * 0.08):
            return False
        return True

    def _draw_detections(
        self,
        frame: np.ndarray,
        person_boxes: list[DetectionBox],
        phone_boxes: list[DetectionBox],
    ) -> None:
        for index, person_box in enumerate(person_boxes, start=1):
            top_left = (int(person_box.x1), int(person_box.y1))
            bottom_right = (int(person_box.x2), int(person_box.y2))
            cv2.rectangle(frame, top_left, bottom_right, (80, 180, 255), 2)
            cv2.putText(
                frame,
                f"student_{index}",
                (top_left[0], max(20, top_left[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (80, 180, 255),
                2,
                cv2.LINE_AA,
            )

        for phone_box in phone_boxes:
            top_left = (int(phone_box.x1), int(phone_box.y1))
            bottom_right = (int(phone_box.x2), int(phone_box.y2))
            cv2.rectangle(frame, top_left, bottom_right, (255, 90, 90), 2)
            cv2.putText(
                frame,
                "cell phone",
                (top_left[0], max(20, top_left[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 90, 90),
                2,
                cv2.LINE_AA,
            )


class PhoneDetectionOrchestrator:
    def __init__(self, detector: YoloPhoneDetector) -> None:
        self.detector = detector

    def detect_classroom(self, frame: np.ndarray) -> ClassroomDetections:
        return self.detector.detect_classroom_objects(frame)

    def detect_and_assign(
        self,
        frame: np.ndarray,
        seat_map: list[StudentSeat],
        source_name: str,
    ) -> tuple[list[PhoneDetectionEvent], np.ndarray]:
        detections = self.detect_classroom(frame)
        if not detections.person_boxes or not seat_map:
            return [], detections.annotated_frame
        phone_assignments = self.assign_phones_to_people(detections.person_boxes, detections.phone_boxes)
        rows = []
        for index, student in enumerate(seat_map[: len(phone_assignments)]):
            rows.append(
                {
                    "seat_id": student.seat_id,
                    "student_name": student.student_name,
                    "roll_number": student.roll_number,
                    "phone_boxes": phone_assignments[index],
                }
            )
        return self.create_phone_events(source_name=source_name, student_rows=rows), detections.annotated_frame

    def create_phone_events(
        self,
        source_name: str,
        student_rows: list[dict],
    ) -> list[PhoneDetectionEvent]:
        events: list[PhoneDetectionEvent] = []
        for student in student_rows:
            for phone_box in student.get("phone_boxes", []):
                events.append(
                    PhoneDetectionEvent(
                        timestamp=datetime.now(),
                        seat_id=student["seat_id"],
                        student_name=student["student_name"],
                        roll_number=student["roll_number"],
                        confidence=round(phone_box.confidence, 4),
                        x1=round(phone_box.x1, 2),
                        y1=round(phone_box.y1, 2),
                        x2=round(phone_box.x2, 2),
                        y2=round(phone_box.y2, 2),
                        source_name=source_name,
                    )
                )
        return events

    def assign_phones_to_people(
        self,
        person_boxes: list[DetectionBox],
        phone_boxes: list[DetectionBox],
    ) -> list[list[DetectionBox]]:
        assigned_phones: list[list[DetectionBox]] = [[] for _ in person_boxes]
        for phone_box in phone_boxes:
            best_index = self._find_matching_person(phone_box, person_boxes)
            if best_index is not None:
                assigned_phones[best_index].append(phone_box)
        return assigned_phones

    def _find_matching_person(
        self,
        phone_box: DetectionBox,
        person_boxes: list[DetectionBox],
    ) -> int | None:
        best_index = None
        best_score = None
        for index, person_box in enumerate(person_boxes):
            if self._is_box_inside(phone_box, person_box):
                score = self._center_distance(phone_box, person_box)
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = index
        if best_index is not None:
            return best_index

        for index, person_box in enumerate(person_boxes):
            score = self._center_distance(phone_box, person_box)
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        return best_index

    def _is_box_inside(self, inner_box: DetectionBox, outer_box: DetectionBox) -> bool:
        margin_x = outer_box.width * 0.08
        margin_y = outer_box.height * 0.12
        return (
            inner_box.center[0] >= outer_box.x1 - margin_x
            and inner_box.center[0] <= outer_box.x2 + margin_x
            and inner_box.center[1] >= outer_box.y1 - margin_y
            and inner_box.center[1] <= outer_box.y2 + margin_y
        )

    def _center_distance(self, phone_box: DetectionBox, person_box: DetectionBox) -> float:
        phone_x, phone_y = phone_box.center
        person_x, person_y = person_box.center
        return ((phone_x - person_x) ** 2 + (phone_y - person_y) ** 2) ** 0.5
