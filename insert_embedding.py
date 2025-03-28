import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------------
# ✅ Load environment variables
# ----------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ----------------------------------
# ✅ Connect to Postgres database
# ----------------------------------
conn = psycopg2.connect(
    dbname="rag",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# ----------------------------------
# ✅ Sample input
# ----------------------------------
title = "Intro to Perovskites"
summary = "A basic explanation of perovskite use"
chunk = "Perovskites are used in solar cells due to their excellent light absorption properties."

# ----------------------------------
# ✅ Generate embedding from OpenAI
# ----------------------------------
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=chunk
)
embedding = response.data[0].embedding

# ----------------------------------
# ✅ Insert into database
# ----------------------------------
cur.execute("""
    INSERT INTO papers (title, summary, chunk, embedding)
    VALUES (%s, %s, %s, %s);
""", (title, summary, chunk, embedding))

conn.commit()
print("✅ Embedding inserted into 'papers' table.")

# ----------------------------------
# ✅ Clean up
# ----------------------------------
cur.close()
conn.close()
