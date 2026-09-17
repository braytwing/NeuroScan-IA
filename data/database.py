import sqlite3
from datetime import datetime
import os

class DatabaseManager:
    def __init__(self, db_name="neuroscan.db"):
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), db_name)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.create_tables()
        self._migrate_database()

    def create_tables(self):
        # Crear estructura básica
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                patient_id TEXT PRIMARY KEY,
                hta BOOLEAN,
                dm2 BOOLEAN,
                anticoag BOOLEAN,
                created_at TIMESTAMP
            )
        ''')
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                face_score REAL,
                arm_score REAL,
                speech_score REAL,
                is_alert BOOLEAN,
                eval_date TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
            )
        ''')
        self.conn.commit()

    def _migrate_database(self):
        """Añade columnas nuevas a bases de datos antiguas sin romper el sistema."""
        columnas_nuevas = [
            ("name", "TEXT"),
            ("age", "INTEGER"),
            ("otros", "TEXT"),
            ("nihss_score", "INTEGER"),
            ("photo_path", "TEXT")
        ]
        for col, tipo in columnas_nuevas:
            try:
                self.cursor.execute(f"ALTER TABLE patients ADD COLUMN {col} {tipo}")
            except sqlite3.OperationalError:
                pass # La columna ya existe
        self.conn.commit()

    def save_patient(self, pid, name, age, hta, dm2, anticoag, otros, nihss=0, photo=""):
        try:
            self.cursor.execute('''
                INSERT OR REPLACE INTO patients (patient_id, name, age, hta, dm2, anticoag, otros, nihss_score, photo_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pid, name, age, hta, dm2, anticoag, otros, nihss, photo, datetime.now()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error BD al salvar paciente: {e}")
            return False

    def save_evaluation(self, patient_id, face_score, arm_score, speech_score, is_alert):
        try:
            self.cursor.execute('''
                INSERT INTO evaluations (patient_id, face_score, arm_score, speech_score, is_alert, eval_date)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (patient_id, face_score, arm_score, speech_score, is_alert, datetime.now()))
            self.conn.commit()
            return True
        except Exception as e:
            print(f"Error BD al salvar evaluación: {e}")
            return False

    def close(self):
        self.conn.close()