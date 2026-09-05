import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "patients")
DB_PATH = os.path.join(DATA_DIR, "patients.db")

TABLES = ["patients", "visits", "conditions", "medications", "procedures"]

if os.path.exists(DB_PATH):
    os.remove(DB_PATH)

conn = sqlite3.connect(DB_PATH)

for table in TABLES:
    df = pd.read_csv(os.path.join(DATA_DIR, f"{table}.csv"))
    df.to_sql(table, conn, index=False)
    print(f"loaded {table}: {len(df)} rows")

print(f"\nSQLite database written to {DB_PATH}")

# --- a couple of sanity-check queries to prove it's really relational now ---
print("\n--- sanity check: visits per patient ---")
for row in conn.execute(
    """
    SELECT patients.FIRST, patients.LAST, COUNT(*) AS visit_count
    FROM visits
    JOIN patients ON visits.PATIENT_ID = patients.PATIENT_ID
    GROUP BY patients.PATIENT_ID
    ORDER BY visit_count DESC
    LIMIT 5
    """
):
    print(row)

print("\n--- sanity check: patients on insulin ---")
for row in conn.execute(
    """
    SELECT DISTINCT patients.FIRST, patients.LAST
    FROM medications
    JOIN visits ON medications.VISIT_ID = visits.VISIT_ID
    JOIN patients ON visits.PATIENT_ID = patients.PATIENT_ID
    WHERE medications.DESCRIPTION LIKE '%insulin%'
    """
):
    print(row)

conn.close()
