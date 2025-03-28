import os
from dotenv import load_dotenv
from comet_ml import Experiment
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import psycopg2

# ✅ Load .env variables
load_dotenv()
print("Loaded COMET_API_KEY:", os.getenv("COMET_API_KEY"))

# ✅ Set up API keys and clients
openai_api_key = os.getenv("OPENAI_API_KEY")
comet_api_key = os.getenv("COMET_API_KEY")
comet_workspace = os.getenv("COMET_WORKSPACE")
comet_project = os.getenv("COMET_PROJECT_NAME")

client = OpenAI(api_key=openai_api_key)

# ✅ Counter to create unique experiment names (RAG Query #1, #2, etc.)
query_counter = 1

# ✅ FastAPI setup
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "running"}

class Question(BaseModel):
    question: str

@app.post("/generate")
def generate_answer(data: Question):
    global query_counter

    question = data.question

    # ✅ Create a new Comet experiment
    experiment = Experiment(
        api_key=comet_api_key,
        workspace=comet_workspace,
        project_name=comet_project,
        auto_output_logging="simple"
    )

    # ✅ Set a unique RAG-style name
    experiment.set_name(f"RAG Query #{query_counter}")
    query_counter += 1  # increment for next run

    experiment.log_text(f"🧠 Question: {question}")

    # ✅ Embed the question
    embedding = client.embeddings.create(
        input=question,
        model="text-embedding-ada-002"
    ).data[0].embedding

    # ✅ Connect to database and fetch top 3 chunks
    conn = psycopg2.connect(
        dbname="rag",
        user="postgres",
        password="postgres",
        host="localhost",
        port=5432
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT chunk
        FROM papers
        ORDER BY embedding <-> %s::vector
        LIMIT 3;
    """, (embedding,))
    results = cur.fetchall()
    cur.close()
    conn.close()

    context = "\n\n".join([row[0] for row in results])
    experiment.log_text(f"📚 Retrieved Chunks:\n{context}")

    # ✅ Generate answer with context
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Use the context below to answer the question."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
    )
    answer = response.choices[0].message.content
    experiment.log_text(f"💬 Answer: {answer}")

    # ✅ Optional: End experiment (clean close)
    experiment.end()

    return {
        "question": question,
        "answer": answer
    }
