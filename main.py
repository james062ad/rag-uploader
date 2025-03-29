# main.py — FastAPI backend with /generate logging
from fastapi import FastAPI
from pydantic import BaseModel
import os
from openai import OpenAI
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

# Set up FastAPI
app = FastAPI()

# OpenAI and Supabase clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# Define request schema
class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate_answer(data: Question):
    try:
        question = data.question
        print("🧠 Received question:", question)

        # Step 1: Embed question
        embedding = client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"
        ).data[0].embedding

        print("✅ Created embedding")

        # Step 2: Retrieve top 3 similar chunks from Supabase
        response = supabase.rpc("match_papers", {
            "query_embedding": embedding,
            "match_count": 3
        }).execute()

        print("📥 Supabase response:", response)

        if not response.data:
            return {"answer": "❌ No matching context found."}

        # Step 3: Combine chunks
        context = "\n\n".join([doc['chunk'] for doc in response.data])

        # Step 4: Send to OpenAI
        prompt = f"""Answer the question based on the following documents:\n\n{context}\n\nQ: {question}\nA:"""

        chat_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = chat_response.choices[0].message.content
        print("✅ Answer generated")

        return {
            "question": question,
            "answer": answer
        }

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}
