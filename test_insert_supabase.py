import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

# ✅ Load environment variables
load_dotenv()

# ✅ Read Supabase & OpenAI credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Connect to Supabase and OpenAI
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Sample chunk to insert
chunk_text = "Perovskites are used in solar cells due to their excellent light absorption properties."
title = "Test Insert"

# ✅ Generate embedding
embedding = client.embeddings.create(
    input=chunk_text,
    model="text-embedding-ada-002"
).data[0].embedding

# ✅ Insert into Supabase
response = supabase.table("papers").insert({
    "title": title,
    "chunk": chunk_text,
    "embedding": embedding
}).execute()

print("✅ Inserted into Supabase!")
print(response)
