import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------
# ✅ Load environment variables
# ---------------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ---------------------------------------
# ✅ Connect to PostgreSQL
# ---------------------------------------
conn = psycopg2.connect(
    dbname="rag",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5432
)
cur = conn.cursor()

# ---------------------------------------
# ✅ User question to search for
# ---------------------------------------
user_question = "What are perovskites used for?"

# ---------------------------------------
# ✅ Generate embedding for the question
# ---------------------------------------
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=user_question
)
query_embedding = response.data[0].embedding

# ---------------------------------------
# ✅ Perform vector similarity search (with type cast)
# ---------------------------------------
cur.execute("""
    SELECT title, summary, chunk
    FROM papers
    ORDER BY embedding <-> %s::vector
    LIMIT 5;
""", (query_embedding,))

results = cur.fetchall()

# ---------------------------------------
# ✅ Display results
# ---------------------------------------
print(f"\n🔍 Top 5 results for: '{user_question}'\n")
for idx, (title, summary, chunk) in enumerate(results, start=1):
    print(f"{idx}. 📘 Title: {title}")
    print(f"   📝 Chunk: {chunk}\n")

# ---------------------------------------
# ✅ Close connection
# ---------------------------------------
cur.close()
conn.close()
