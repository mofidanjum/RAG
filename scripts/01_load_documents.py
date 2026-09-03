import csv
import os

from langchain_community.document_loaders import TextLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXT_DIR = os.path.join(BASE_DIR, "data", "txt")
COMPLAINTS_CSV = os.path.join(BASE_DIR, "data", "complaints.csv")
COMPANIES_CSV = os.path.join(BASE_DIR, "data", "companies.csv")

# --- load the structured side into lookup dicts, keyed by id ---
companies_by_id = {}
with open(COMPANIES_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        companies_by_id[row["company_id"]] = row["company_name"]

complaints_by_id = {}
with open(COMPLAINTS_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        complaints_by_id[row["complaint_id"]] = row

# --- load every text file as a LangChain Document, attach structured metadata ---
documents = []
for filename in os.listdir(TXT_DIR):
    complaint_id = filename.removeprefix("complaint_").removesuffix(".txt")
    complaint_row = complaints_by_id[complaint_id]
    company_name = companies_by_id[complaint_row["company_id"]]

    loader = TextLoader(os.path.join(TXT_DIR, filename), encoding="utf-8")
    doc = loader.load()[0]  # TextLoader always returns a list of exactly one Document

    doc.metadata["complaint_id"] = complaint_id
    doc.metadata["company"] = company_name
    doc.metadata["product"] = complaint_row["product"]
    doc.metadata["issue"] = complaint_row["issue"]
    doc.metadata["state"] = complaint_row["state"]

    documents.append(doc)

print(f"Loaded {len(documents)} documents\n")

sample = documents[0]
print("--- sample Document ---")
print("metadata:", sample.metadata)
print("page_content (first 300 chars):")
print(sample.page_content[:300])
