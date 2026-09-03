import csv
import os
from collections import Counter, defaultdict

import pandas as pd

TOP_N_COMPANIES = 6    # keep only the most-complained-about companies
MAX_PER_COMPANY = 40   # cap so one company doesn't dominate the file count

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "data", "complaints_full.parquet")
TXT_DIR = os.path.join(BASE_DIR, "data", "txt")
COMPLAINTS_CSV = os.path.join(BASE_DIR, "data", "complaints.csv")
COMPANIES_CSV = os.path.join(BASE_DIR, "data", "companies.csv")

os.makedirs(TXT_DIR, exist_ok=True)

print("loading parquet file...")
df = pd.read_parquet(PARQUET_PATH)
print(f"loaded {len(df)} total rows")

# keep only rows with a real narrative
df = df[df["Consumer Complaint"].notna()]
df = df[df["Consumer Complaint"].str.strip().str.len() > 50]
print(f"{len(df)} rows have usable narrative text")

# keep only the top-N most-complained-about companies
top_companies = df["Company"].value_counts().head(TOP_N_COMPANIES).index.tolist()
df = df[df["Company"].isin(top_companies)]

# cap per company
df = df.groupby("Company", group_keys=False).head(MAX_PER_COMPANY)
collected = df.to_dict("records")

# companies.csv: one row per unique company
company_ids = {name: i + 1 for i, name in enumerate(sorted(top_companies))}

with open(COMPANIES_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["company_id", "company_name"])
    for name, cid in company_ids.items():
        writer.writerow([cid, name])

# complaints.csv: fact table, links to companies.csv via company_id
with open(COMPLAINTS_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complaint_id", "company_id", "product", "issue", "state",
        "date_received", "company_response", "timely_response",
    ])
    for row in collected:
        writer.writerow([
            row["Complaint ID"],
            company_ids[row["Company"]],
            row["Product"],
            row["Issue"],
            row["State"],
            row["Date received"],
            row["Company Response to Consumer"],
            row["Timely response?"],
        ])

# unstructured narrative, one file per complaint, named by complaint_id
for row in collected:
    path = os.path.join(TXT_DIR, f"complaint_{row['Complaint ID']}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(row["Consumer Complaint"])

print(f"\nDone: {len(collected)} complaints across {len(top_companies)} companies")
print(f"  {COMPANIES_CSV}")
print(f"  {COMPLAINTS_CSV}")
print(f"  {TXT_DIR}\\complaint_<id>.txt")

print("\n--- company breakdown (for picking the multi-hop demo question) ---")
by_company_issues = defaultdict(Counter)
company_counts = Counter()
for row in collected:
    by_company_issues[row["Company"]][row["Issue"]] += 1
    company_counts[row["Company"]] += 1

for name, total in company_counts.most_common():
    issues = by_company_issues[name]
    print(f"\n{name}: {total} complaints, {len(issues)} distinct issues")
    for issue, cnt in issues.most_common(5):
        print(f"    {cnt:>3}  {issue}")
