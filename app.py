# --------------------------------------
# ✅ Imports
# --------------------------------------
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
import fitz  # PyMuPDF
from comet_ml import Experiment
import requests

# --------------------------------------
# ✅ Load environment variables
# --------------------------------------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
COMET_PROJECT = os.getenv("COMET_PROJECT_NAME")
FASTAPI_URL = "https://rag-uploader.onrender.com/generate"

# --------------------------------------
# ✅ Initialize clients
# --------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------
# ✅ Streamlit Page
# --------------------------------------
st.set_page_config(page_title="RAG System", layout="centered")
st.title("📄 Upload and Ask Questions")

# -------------------------------
# 📤 Upload Interface
# -------------------------------
st.header("📤 Upload and Embed Document")
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
chunk_size = st.slider("Chunk size (in characters)", 300, 1000, 500, 100)

# Extract text
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "".join([page.get_text() for page in doc])

if uploaded_file:
    filename = uploaded_file.name
    filetype = filename.split(".")[-1].lower()

    if filetype == "txt":
        file_text = uploaded_file.read().decode("utf-8")
    elif filetype == "pdf":
        file_text = extract_text_from_pdf(uploaded_file.read())
    else:
        st.error("Unsupported file type.")
        st.stop()

    chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]
    st.info(f"📦 Found {len(chunks)} chunks to embed and store.")

    if st.button("Upload to Supabase"):
        with st.spinner("Embedding and uploading..."):
            progress = st.progress(0)

            experiment = Experiment(
                api_key=COMET_API_KEY,
                workspace=COMET_WORKSPACE,
                project_name=COMET_PROJECT,
                auto_output_logging="simple"
            )
            experiment.set_name(f"Upload: {filename} ({len(chunks)} chunks)")
            experiment.log_text("First chunk preview:\n" + chunks[0])

            for i, chunk in enumerate(chunks):
                try:
                    embedding = client.embeddings.create(
                        input=chunk,
                        model="text-embedding-ada-002"
                    ).data[0].embedding

                    supabase.table("papers").insert({
                        "title": f"Upload: {filename}",
                        "chunk": chunk,
                        "embedding": embedding
                    }).execute()

                    progress.progress((i + 1) / len(chunks))

                except Exception as e:
                    st.error(f"❌ Failed to insert chunk {i+1}: {e}")
                    experiment.log_other(f"Chunk {i+1} failed", str(e))

            experiment.end()
            st.success(f"✅ Uploaded {len(chunks)} chunks to Supabase and logged to Comet!")

# -------------------------------
# 🤖 QA Interface
# -------------------------------
st.header("💬 Ask a Question")
question = st.text_input("What would you like to know?")
if st.button("Generate Answer"):
    with st.spinner("🧠 Generating answer..."):
        try:
            response = requests.post(FASTAPI_URL, json={"question": question})
            result = response.json()
            st.success("Answer:")
            st.write(result.get("answer", "❌ No answer found."))
        except Exception as e:
            st.error(f"Request failed: {e}")
