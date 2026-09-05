import os

import certifi
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustCustomCAs

load_dotenv()

# neo4j+s:// bakes in "verify against the OS trust store," which is what's
# failing on this machine (Windows hasn't cached the SSL.com root CA). Using
# the bare neo4j:// scheme lets us instead verify against certifi's bundle,
# which ships a complete, current CA list independent of the OS.
uri = os.environ["NEO4J_URI"].replace("neo4j+s://", "neo4j://")
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

driver = GraphDatabase.driver(
    uri,
    auth=(username, password),
    encrypted=True,
    trusted_certificates=TrustCustomCAs(certifi.where()),
)

with driver.session() as session:
    result = session.run("RETURN 'connected!' AS message")
    print(result.single()["message"])

driver.close()
