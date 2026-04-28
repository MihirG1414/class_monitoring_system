import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


class CSVReportGenerator:
    def __init__(self, reports_directory: Path | None = None) -> None:
        self.reports_directory = reports_directory or Path("reports")
        self.reports_directory.mkdir(parents=True, exist_ok=True)

    def build_session_report(
        self,
        attention_events: pd.DataFrame,
        phone_events: pd.DataFrame,
        seat_map: list,
    ) -> pd.DataFrame:
        if seat_map:
            base_rows = [
                {
                    "seat_id": student.seat_id,
                    "roll_number": student.roll_number,
                    "name": student.student_name,
                }
                for student in seat_map
            ]
        elif not attention_events.empty:
            base_rows = (
                attention_events[["seat_id", "roll_number", "student_name"]]
                .drop_duplicates(subset=["seat_id"])
                .rename(columns={"student_name": "name"})
                .to_dict(orient="records")
            )
        else:
            base_rows = []
        report_dataframe = pd.DataFrame(base_rows)
        if attention_events.empty:
            report_dataframe["total_session_duration"] = 0
            report_dataframe["attentive_duration"] = 0
            report_dataframe["distracted_duration"] = 0
            report_dataframe["face_visible"] = "Unknown"
            report_dataframe["phone_visible"] = "No"
            report_dataframe["phone_detection_count"] = 0
            report_dataframe["alert_count"] = 0
            report_dataframe["attention_percentage"] = 0.0
            report_dataframe["final_remark"] = "No monitoring data"
            return report_dataframe

        attention_events = attention_events.copy()
        attention_events["timestamp"] = pd.to_datetime(attention_events["timestamp"])

        session_rows: list[dict] = []
        for student in seat_map:
            student_events = attention_events[attention_events["seat_id"] == student.seat_id].sort_values("timestamp")
            if student_events.empty:
                session_rows.append(
                    {
                        "seat_id": student.seat_id,
                        "total_session_duration": 0,
                        "attentive_duration": 0,
                        "distracted_duration": 0,
                        "alert_count": 0,
                        "attention_percentage": 0.0,
                        "face_visible": "Unknown",
                    }
                )
                continue

            durations = self._estimate_event_durations(student_events)
            total_session_duration = int(round(sum(durations)))
            attentive_duration = int(
                round(
                    sum(
                        duration
                        for duration, status in zip(durations, student_events["status"])
                        if status == "Attentive"
                    )
                )
            )
            distracted_duration = int(
                round(
                    sum(
                        duration
                        for duration, status in zip(durations, student_events["status"])
                        if status != "Attentive"
                    )
                )
            )
            final_alert_count = int(student_events["alert_count"].iloc[-1])
            attention_percentage = round(
                (attentive_duration / total_session_duration) * 100 if total_session_duration else 0.0,
                2,
            )
            latest_head_pose = str(student_events["head_pose"].iloc[-1])
            session_rows.append(
                {
                    "seat_id": student.seat_id,
                    "total_session_duration": total_session_duration,
                    "attentive_duration": attentive_duration,
                    "distracted_duration": distracted_duration,
                    "alert_count": final_alert_count,
                    "attention_percentage": attention_percentage,
                    "face_visible": "No" if latest_head_pose == "FACE_NOT_VISIBLE" else "Yes",
                }
            )

        session_dataframe = pd.DataFrame(session_rows)
        report_dataframe = report_dataframe.merge(session_dataframe, on="seat_id", how="left")

        if phone_events.empty:
            phone_counts = pd.DataFrame(columns=["seat_id", "phone_detection_count"])
        else:
            phone_counts = (
                phone_events.groupby("seat_id")
                .size()
                .reset_index(name="phone_detection_count")
            )
        report_dataframe = report_dataframe.merge(phone_counts, on="seat_id", how="left")
        report_dataframe["phone_detection_count"] = pd.to_numeric(
            report_dataframe["phone_detection_count"],
            errors="coerce",
        ).fillna(0).astype(int)
        report_dataframe["phone_visible"] = report_dataframe["phone_detection_count"].apply(lambda count: "Yes" if count > 0 else "No")
        report_dataframe["final_remark"] = report_dataframe.apply(self._build_final_remark, axis=1)
        return report_dataframe[
            [
                "roll_number",
                "name",
                "total_session_duration",
                "attentive_duration",
                "distracted_duration",
                "face_visible",
                "phone_visible",
                "phone_detection_count",
                "alert_count",
                "attention_percentage",
                "final_remark",
            ]
        ]

    def _estimate_event_durations(self, student_events: pd.DataFrame) -> list[float]:
        timestamps = list(student_events["timestamp"])
        if len(timestamps) == 1:
            return [1.0]

        durations: list[float] = []
        for index in range(len(timestamps) - 1):
            delta_seconds = (timestamps[index + 1] - timestamps[index]).total_seconds()
            durations.append(max(1.0, delta_seconds))
        last_known_duration = durations[-1] if durations else 1.0
        durations.append(last_known_duration)
        return durations

    def _build_final_remark(self, row: pd.Series) -> str:
        if row["attention_percentage"] >= 85 and row["phone_detection_count"] == 0:
            return "Highly attentive"
        if row["attention_percentage"] >= 70 and row["alert_count"] <= 1:
            return "Satisfactory attention"
        if row["phone_detection_count"] > 0 or row["alert_count"] > 1:
            return "Needs attention improvement"
        return "Monitor more closely"

    def generate(self, report_dataframe: pd.DataFrame) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_directory / f"classroom_attention_report_{timestamp}.csv"
        fieldnames = [
            "roll_number",
            "name",
            "total_session_duration",
            "attentive_duration",
            "distracted_duration",
            "face_visible",
            "phone_visible",
            "phone_detection_count",
            "alert_count",
            "attention_percentage",
            "final_remark",
        ]
        with report_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report_dataframe.to_dict(orient="records"))
        return report_path
