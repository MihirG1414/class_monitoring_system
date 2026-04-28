import sqlite3
from pathlib import Path

import pandas as pd

from src.models import AttentionEvent, PhoneDetectionEvent


class SQLiteEventRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize_database(self) -> None:
        with self._get_connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    seat_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attention_percentage REAL NOT NULL,
                    distracted_duration_seconds INTEGER NOT NULL,
                    alert_count INTEGER NOT NULL,
                    head_pose TEXT NOT NULL DEFAULT 'SIMULATED',
                    source_name TEXT NOT NULL DEFAULT 'simulation'
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS phone_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    seat_id TEXT NOT NULL,
                    student_name TEXT NOT NULL,
                    roll_number TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    x1 REAL NOT NULL,
                    y1 REAL NOT NULL,
                    x2 REAL NOT NULL,
                    y2 REAL NOT NULL,
                    source_name TEXT NOT NULL
                )
                """
            )
            self._ensure_attention_event_columns(connection)
            connection.commit()

    def _ensure_attention_event_columns(self, connection: sqlite3.Connection) -> None:
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(attention_events)").fetchall()
        }
        if "head_pose" not in existing_columns:
            connection.execute(
                "ALTER TABLE attention_events ADD COLUMN head_pose TEXT NOT NULL DEFAULT 'SIMULATED'"
            )
        if "source_name" not in existing_columns:
            connection.execute(
                "ALTER TABLE attention_events ADD COLUMN source_name TEXT NOT NULL DEFAULT 'simulation'"
            )

    def save_events(self, events: list[AttentionEvent]) -> None:
        with self._get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO attention_events (
                    timestamp,
                    seat_id,
                    student_name,
                    roll_number,
                    status,
                    attention_percentage,
                    distracted_duration_seconds,
                    alert_count,
                    head_pose,
                    source_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.timestamp.isoformat(),
                        event.seat_id,
                        event.student_name,
                        event.roll_number,
                        event.status,
                        event.attention_percentage,
                        event.distracted_duration_seconds,
                        event.alert_count,
                        event.head_pose,
                        event.source_name,
                    )
                    for event in events
                ],
            )
            connection.commit()

    def save_phone_events(self, events: list[PhoneDetectionEvent]) -> None:
        if not events:
            return
        with self._get_connection() as connection:
            connection.executemany(
                """
                INSERT INTO phone_events (
                    timestamp,
                    seat_id,
                    student_name,
                    roll_number,
                    confidence,
                    x1,
                    y1,
                    x2,
                    y2,
                    source_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.timestamp.isoformat(),
                        event.seat_id,
                        event.student_name,
                        event.roll_number,
                        event.confidence,
                        event.x1,
                        event.y1,
                        event.x2,
                        event.y2,
                        event.source_name,
                    )
                    for event in events
                ],
            )
            connection.commit()

    def fetch_events(self) -> pd.DataFrame:
        with self._get_connection() as connection:
            dataframe = pd.read_sql_query(
                "SELECT * FROM attention_events ORDER BY timestamp DESC",
                connection,
            )
        return dataframe

    def fetch_phone_events(self) -> pd.DataFrame:
        with self._get_connection() as connection:
            dataframe = pd.read_sql_query(
                "SELECT * FROM phone_events ORDER BY timestamp DESC",
                connection,
            )
        return dataframe

    def fetch_phone_detection_counts(self) -> dict[str, int]:
        with self._get_connection() as connection:
            cursor = connection.execute(
                """
                SELECT seat_id, COUNT(*) AS detection_count
                FROM phone_events
                GROUP BY seat_id
                """
            )
            rows = cursor.fetchall()
        return {seat_id: int(count) for seat_id, count in rows}

    def fetch_recent_events(self, limit: int = 15) -> pd.DataFrame:
        with self._get_connection() as connection:
            attention_events = pd.read_sql_query(
                """
                SELECT timestamp, seat_id, student_name, roll_number, 'attention' AS event_type,
                       status, attention_percentage, distracted_duration_seconds, alert_count,
                       NULL AS confidence, head_pose, source_name
                FROM attention_events
                LIMIT ?
                """,
                connection,
                params=(limit,),
            )
            phone_events = pd.read_sql_query(
                """
                SELECT timestamp, seat_id, student_name, roll_number, 'phone' AS event_type,
                       'Phone Detected' AS status, NULL AS attention_percentage,
                       NULL AS distracted_duration_seconds, NULL AS alert_count,
                       confidence, NULL AS head_pose, source_name
                FROM phone_events
                LIMIT ?
                """,
                connection,
                params=(limit,),
            )
        non_empty_frames = [frame for frame in (attention_events, phone_events) if not frame.empty]
        if not non_empty_frames:
            return pd.DataFrame()
        dataframe = pd.concat(non_empty_frames, ignore_index=True)
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
        dataframe = dataframe.sort_values("timestamp", ascending=False).head(limit)
        dataframe["timestamp"] = dataframe["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return dataframe.reset_index(drop=True)

    def clear(self) -> None:
        with self._get_connection() as connection:
            connection.execute("DELETE FROM attention_events")
            connection.execute("DELETE FROM phone_events")
            connection.commit()
