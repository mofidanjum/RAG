import os

import pandas as pd

N_PATIENTS = 15
MAX_VISITS_PER_PATIENT = 20

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_DIR = os.path.join(BASE_DIR, "data", "patients", "csv")
OUT_DIR = os.path.join(BASE_DIR, "data", "patients")
NOTES_DIR = os.path.join(OUT_DIR, "notes")

os.makedirs(NOTES_DIR, exist_ok=True)

print("loading source tables...")
patients = pd.read_csv(os.path.join(CSV_DIR, "patients.csv"))
visits = pd.read_csv(os.path.join(CSV_DIR, "encounters.csv"))
conditions = pd.read_csv(os.path.join(CSV_DIR, "conditions.csv"))
medications = pd.read_csv(os.path.join(CSV_DIR, "medications.csv"))
procedures = pd.read_csv(os.path.join(CSV_DIR, "procedures.csv"))

# rename cryptic source-file column names to plain, human-readable ones
patients = patients.rename(columns={"Id": "PATIENT_ID"})
visits = visits.rename(columns={"Id": "VISIT_ID", "PATIENT": "PATIENT_ID"})
conditions = conditions.rename(columns={"ENCOUNTER": "VISIT_ID"})
medications = medications.rename(columns={"ENCOUNTER": "VISIT_ID"})
procedures = procedures.rename(columns={"ENCOUNTER": "VISIT_ID"})

# --- pick patients with rich-enough histories to make relationships interesting ---
visit_counts = visits.groupby("PATIENT_ID").size().sort_values(ascending=False)
selected_patient_ids = visit_counts[visit_counts >= 10].head(N_PATIENTS).index.tolist()

patients = patients[patients["PATIENT_ID"].isin(selected_patient_ids)].copy()

# cap visits per patient so one patient can't dominate
visits = visits[visits["PATIENT_ID"].isin(selected_patient_ids)]
visits = visits.groupby("PATIENT_ID", group_keys=False).head(MAX_VISITS_PER_PATIENT)
selected_visit_ids = set(visits["VISIT_ID"])

conditions = conditions[conditions["VISIT_ID"].isin(selected_visit_ids)].copy()
medications = medications[medications["VISIT_ID"].isin(selected_visit_ids)].copy()
procedures = procedures[procedures["VISIT_ID"].isin(selected_visit_ids)].copy()

# --- write the filtered, still-relational structured tables ---
patients.to_csv(os.path.join(OUT_DIR, "patients.csv"), index=False)
visits.to_csv(os.path.join(OUT_DIR, "visits.csv"), index=False)
conditions.to_csv(os.path.join(OUT_DIR, "conditions.csv"), index=False)
medications.to_csv(os.path.join(OUT_DIR, "medications.csv"), index=False)
procedures.to_csv(os.path.join(OUT_DIR, "procedures.csv"), index=False)

# --- generate one unstructured note per visit, derived from the structured rows ---
patient_name = {
    row["PATIENT_ID"]: f"{row['FIRST']} {row['LAST']}" for _, row in patients.iterrows()
}

for _, visit in visits.iterrows():
    visit_id = visit["VISIT_ID"]
    name = patient_name[visit["PATIENT_ID"]]

    visit_conditions = conditions[conditions["VISIT_ID"] == visit_id]["DESCRIPTION"].tolist()
    visit_medications = medications[medications["VISIT_ID"] == visit_id]["DESCRIPTION"].tolist()
    visit_procedures = procedures[procedures["VISIT_ID"] == visit_id]["DESCRIPTION"].tolist()

    lines = [
        f"Patient: {name}",
        f"Visit date: {visit['START']}",
        f"Visit type: {visit['DESCRIPTION']} ({visit['ENCOUNTERCLASS']})",
    ]
    if pd.notna(visit.get("REASONDESCRIPTION")):
        lines.append(f"Reason for visit: {visit['REASONDESCRIPTION']}")
    if visit_conditions:
        lines.append(f"Conditions noted: {', '.join(visit_conditions)}")
    if visit_medications:
        lines.append(f"Medications prescribed: {', '.join(visit_medications)}")
    if visit_procedures:
        lines.append(f"Procedures performed: {', '.join(visit_procedures)}")

    note_text = "\n".join(lines)
    with open(os.path.join(NOTES_DIR, f"note_{visit_id}.txt"), "w", encoding="utf-8") as f:
        f.write(note_text)

print(f"\nDone: {len(patients)} patients, {len(visits)} visits, {len(visits)} notes")
print(f"  {OUT_DIR}\\patients.csv, visits.csv, conditions.csv, medications.csv, procedures.csv")
print(f"  {NOTES_DIR}\\note_<visit_id>.txt")

print("\n--- patient breakdown ---")
for _, p in patients.iterrows():
    pid = p["PATIENT_ID"]
    patient_visit_ids = visits[visits["PATIENT_ID"] == pid]["VISIT_ID"]
    n_visits = len(patient_visit_ids)
    n_cond = conditions[conditions["VISIT_ID"].isin(patient_visit_ids)]["DESCRIPTION"].nunique()
    n_meds = medications[medications["VISIT_ID"].isin(patient_visit_ids)]["DESCRIPTION"].nunique()
    print(f"{p['FIRST']} {p['LAST']}: {n_visits} visits, {n_cond} distinct conditions, {n_meds} distinct medications")
