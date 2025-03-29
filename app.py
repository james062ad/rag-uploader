import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
import fitz  # PyMuPDF
from comet_ml import Experiment

load_dotenv()

# Keys
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
COMET_PROJECT = os.getenv("COMET_PROJECT_NAME")

# Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# UI
st.set_page_config(page_title="Upload to RAG", page_icon="📄")
st.title("📄 Upload and Embed Document into RAG System")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
chunk_size = st.slider("Chunk size (in characters)", 300, 1000, 500, 100)

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return "".join([page.get_text() for page in doc])

if uploaded_file:
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    text = uploaded_file.read()
    file_text = text.decode("utf-8") if ext == "txt" else extract_text_from_pdf(text)

    chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]
    st.markdown(f"📦 Found **{len(chunks)}** chunks to embed and store.")

    if st.button("Upload to Supabase"):
        with st.spinner("🧠 Embedding and uploading..."):
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
                    st.error(f"❌ Failed to insert chunk {i+1}")
                    experiment.log_text(f"Failed chunk {i+1}: {e}")

            experiment.end()
            st.success(f"✅ Uploaded {len(chunks)} chunks to Supabase and logged to Comet!")
