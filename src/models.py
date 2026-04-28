from dataclasses import dataclass
from datetime import datetime


@dataclass
class StudentSeat:
    seat_id: str
    student_name: str
    roll_number: str
    initial_alert_count: int = 0
    zone_x1: float = 0.0
    zone_y1: float = 0.0
    zone_x2: float = 1.0
    zone_y2: float = 1.0

    @property
    def zone_center(self) -> tuple[float, float]:
        return ((self.zone_x1 + self.zone_x2) / 2, (self.zone_y1 + self.zone_y2) / 2)


@dataclass
class AttentionEvent:
    timestamp: datetime
    seat_id: str
    student_name: str
    roll_number: str
    status: str
    attention_percentage: float
    distracted_duration_seconds: int
    alert_count: int
    head_pose: str = "SIMULATED"
    source_name: str = "simulation"


@dataclass
class PhoneDetectionEvent:
    timestamp: datetime
    seat_id: str
    student_name: str
    roll_number: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    source_name: str
