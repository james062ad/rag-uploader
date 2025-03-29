import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
import fitz  # PyMuPDF
from comet_ml import Experiment

# ✅ Load .env
load_dotenv()

# 🔐 Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
COMET_PROJECT = os.getenv("COMET_PROJECT_NAME")

# ✅ Initialize clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ✅ Streamlit UI
st.set_page_config(page_title="Upload to RAG", page_icon="📄")
st.title("📄 Upload and Embed Document into RAG System")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
chunk_size = st.slider("Chunk size (in characters)", 300, 1000, 500, 100)

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

if uploaded_file:
    filename = uploaded_file.name
    filetype = filename.split(".")[-1].lower()

    # ✅ Read file content
    if filetype == "txt":
        file_text = uploaded_file.read().decode("utf-8")
    elif filetype == "pdf":
        file_text = extract_text_from_pdf(uploaded_file.read())
    else:
        st.error("Unsupported file type.")
        st.stop()

    # ✅ Chunk the text
    chunks = [file_text[i:i+chunk_size] for i in range(0, len(file_text), chunk_size)]
    st.write(f"📦 Found {len(chunks)} chunks to embed and store.")

    if st.button("Upload to Supabase"):
        with st.spinner("Embedding and uploading..."):
            progress = st.progress(0)

            # ✅ Create Comet experiment
            experiment = Experiment(
                api_key=COMET_API_KEY,
                workspace=COMET_WORKSPACE,
                project_name=COMET_PROJECT,
                auto_output_logging="simple"
            )
            experiment.set_name(f"Upload: {filename} ({len(chunks)} chunks)")
            experiment.log_text("First chunk preview:\n" + chunks[0])

            for i, chunk in enumerate(chunks):
                # Embed
                embedding = client.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                ).data[0].embedding

                # Insert
                supabase.table("papers").insert({
                    "title": f"Upload: {filename}",
                    "chunk": chunk,
                    "embedding": embedding
                }).execute()

                progress.progress((i + 1) / len(chunks))

            experiment.end()
            st.success(f"✅ Uploaded {len(chunks)} chunks to Supabase and logged to Comet!")
