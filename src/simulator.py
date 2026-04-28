import random
from datetime import datetime
from typing import Iterable

from src.models import AttentionEvent, StudentSeat


class DummyAttentionDetector:
    """Simulates classroom attention states for MVP testing."""

    def __init__(self, seed: int = 42) -> None:
        self._random = random.Random(seed)

    def detect(self, students: Iterable[StudentSeat], previous_alerts: dict[str, int]) -> list[AttentionEvent]:
        events: list[AttentionEvent] = []
        for student in students:
            status = self._random.choices(
                population=["Attentive", "Distracted"],
                weights=[0.7, 0.3],
                k=1,
            )[0]
            attention_percentage = self._random.uniform(72.0, 99.0) if status == "Attentive" else self._random.uniform(25.0, 68.0)
            distracted_duration_seconds = 0 if status == "Attentive" else self._random.randint(20, 180)
            alert_count = previous_alerts.get(student.seat_id, student.initial_alert_count)
            if status == "Distracted":
                alert_count += 1
            events.append(
                AttentionEvent(
                    timestamp=datetime.now(),
                    seat_id=student.seat_id,
                    student_name=student.student_name,
                    roll_number=student.roll_number,
                    status=status,
                    attention_percentage=round(attention_percentage, 2),
                    distracted_duration_seconds=distracted_duration_seconds,
                    alert_count=alert_count,
                )
            )
        return events
