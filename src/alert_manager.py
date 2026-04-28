from __future__ import annotations

import os


class AlertManager:
    """Simulates non-disruptive alert actions for the dashboard."""

    def __init__(self, teacher_phone_number: str = "+918605095251") -> None:
        self.teacher_phone_number = teacher_phone_number

    def trigger_led_alert(self, student_id: str) -> str:
        return f"LED alert simulated for {student_id}"

    def trigger_vibration_alert(self, student_id: str) -> str:
        return f"Vibration alert simulated for {student_id}"

    def send_mqtt_alert(self, student_id: str) -> str:
        return f"MQTT alert simulated for {student_id}"

    def send_sms_alert(self, student_id: str, message: str) -> str:
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        from_number = os.getenv("TWILIO_FROM_NUMBER")
        if not account_sid or not auth_token or not from_number:
            return (
                f"SMS alert simulated for {student_id} -> {self.teacher_phone_number} "
                "(missing Twilio environment variables)"
            )

        try:
            from twilio.rest import Client
        except ImportError:
            return (
                f"SMS alert simulated for {student_id} -> {self.teacher_phone_number} "
                "(Twilio package not available)"
            )

        try:
            client = Client(account_sid, auth_token)
            client.messages.create(
                body=message,
                from_=from_number,
                to=self.teacher_phone_number,
            )
        except Exception as error:
            return (
                f"SMS alert failed for {student_id} -> {self.teacher_phone_number} "
                f"({error})"
            )
        return f"SMS alert sent for {student_id} -> {self.teacher_phone_number}"

    def trigger_all(self, student_id: str) -> list[str]:
        return [
            self.trigger_led_alert(student_id),
            self.trigger_vibration_alert(student_id),
            self.send_mqtt_alert(student_id),
            self.send_sms_alert(
                student_id,
                f"{student_id} appears inattentive in class. Please pay attention.",
            ),
        ]
