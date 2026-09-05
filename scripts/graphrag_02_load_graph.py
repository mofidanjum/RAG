import os

import certifi
import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "patients")

uri = os.environ["NEO4J_URI"].replace("neo4j+s://", "neo4j://")
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(
    uri,
    auth=(username, password),
    encrypted=True,
    trusted_certificates=TrustCustomCAs(certifi.where()),
)

patients = pd.read_csv(os.path.join(DATA_DIR, "patients.csv"))
visits = pd.read_csv(os.path.join(DATA_DIR, "visits.csv"))
conditions = pd.read_csv(os.path.join(DATA_DIR, "conditions.csv"))
medications = pd.read_csv(os.path.join(DATA_DIR, "medications.csv"))
procedures = pd.read_csv(os.path.join(DATA_DIR, "procedures.csv"))


def run(session, query, **params):
    session.run(query, **params)


with driver.session() as session:
    print("clearing any existing graph...")
    session.run("MATCH (n) DETACH DELETE n")

    print("creating uniqueness constraints...")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Visit) REQUIRE v.id IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Medication) REQUIRE m.name IS UNIQUE")
    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (pr:Procedure) REQUIRE pr.name IS UNIQUE")

    print(f"creating {len(patients)} Patient nodes...")
    for _, p in patients.iterrows():
        session.run(
            "MERGE (pat:Patient {id: $id}) SET pat.name = $name, pat.gender = $gender",
            id=p["PATIENT_ID"],
            name=f"{p['FIRST']} {p['LAST']}",
            gender=p["GENDER"],
        )

    print(f"creating {len(visits)} Visit nodes + HAD_VISIT relationships...")
    for _, v in visits.iterrows():
        session.run(
            """
            MATCH (pat:Patient {id: $patient_id})
            MERGE (vis:Visit {id: $visit_id})
            SET vis.date = $date, vis.type = $type
            MERGE (pat)-[:HAD_VISIT]->(vis)
            """,
            patient_id=v["PATIENT_ID"],
            visit_id=v["VISIT_ID"],
            date=v["START"],
            type=v["DESCRIPTION"],
        )

    print(f"creating {len(conditions)} DIAGNOSED_WITH relationships...")
    for _, c in conditions.iterrows():
        session.run(
            """
            MATCH (vis:Visit {id: $visit_id})
            MERGE (cond:Condition {name: $name})
            MERGE (vis)-[:DIAGNOSED_WITH]->(cond)
            """,
            visit_id=c["VISIT_ID"],
            name=c["DESCRIPTION"],
        )

    print(f"creating {len(medications)} PRESCRIBED relationships...")
    for _, m in medications.iterrows():
        session.run(
            """
            MATCH (vis:Visit {id: $visit_id})
            MERGE (med:Medication {name: $name})
            MERGE (vis)-[:PRESCRIBED]->(med)
            """,
            visit_id=m["VISIT_ID"],
            name=m["DESCRIPTION"],
        )

    print(f"creating {len(procedures)} UNDERWENT relationships...")
    for _, pr in procedures.iterrows():
        session.run(
            """
            MATCH (vis:Visit {id: $visit_id})
            MERGE (proc:Procedure {name: $name})
            MERGE (vis)-[:UNDERWENT]->(proc)
            """,
            visit_id=pr["VISIT_ID"],
            name=pr["DESCRIPTION"],
        )

    print("\n--- sanity check: multi-hop query ---")
    print("Patients on insulin AND their other, unrelated conditions:")
    result = session.run(
        """
        MATCH (pat:Patient)-[:HAD_VISIT]->(:Visit)-[:PRESCRIBED]->(med:Medication)
        WHERE med.name CONTAINS 'insulin'
        MATCH (pat)-[:HAD_VISIT]->(:Visit)-[:DIAGNOSED_WITH]->(cond:Condition)
        WHERE NOT cond.name IN ['Diabetes', 'Prediabetes', 'Metabolic syndrome X (disorder)',
                                 'Hypertriglyceridemia (disorder)', 'Diabetic renal disease (disorder)',
                                 'Body mass index 30+ - obesity (finding)']
        RETURN DISTINCT pat.name AS patient, collect(DISTINCT cond.name) AS other_conditions
        """
    )
    for row in result:
        print(f"  {row['patient']}: {row['other_conditions']}")

driver.close()
print("\nDone.")
