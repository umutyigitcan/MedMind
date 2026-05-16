import psycopg2
from psycopg2.extensions import connection

from app.config import settings


def get_connection() -> connection:
    """
    Create and return a PostgreSQL connection.

    The database connection values are loaded from environment variables.
    This keeps credentials outside the source code.
    """
    return psycopg2.connect(
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port,
    )


def create_patients_table() -> None:
    """
    Create the patients table if it does not exist.

    This table stores appointment day and time values extracted from user messages.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id SERIAL PRIMARY KEY,
            day VARCHAR(50) NOT NULL,
            time VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )

    conn.commit()
    cur.close()
    conn.close()


def add_appointment(day: str, time: str) -> None:
    """
    Save an appointment record into PostgreSQL.
    """
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO patients(day, time) VALUES(%s, %s);",
        (day, time),
    )

    conn.commit()
    cur.close()
    conn.close()
