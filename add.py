import sqlite3
from datetime import datetime, timedelta

class DataCleanup:
    """Manage old classroom event data"""
    
    def remove_old_events(self, db_path, days_old=30):
        """Remove events older than specified days"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cursor.execute("DELETE FROM events WHERE timestamp < ?", (cutoff_date,))
        
        conn.commit()
        deleted_rows = cursor.rowcount
        conn.close()
        
        return deleted_rows
