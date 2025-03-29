# --------------------------------------
# ✅ Imports
# --------------------------------------
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
import fitz  # PyMuPDF for PDF parsing
from comet_ml import Experiment

# --------------------------------------
# ✅ Load .env or Streamlit secrets
# --------------------------------------
load_dotenv()

# Secrets for Streamlit Cloud deployment
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COMET_API_KEY = os.getenv("COMET_API_KEY")
COMET_WORKSPACE = os.getenv("COMET_WORKSPACE")
COMET_PROJECT = os.getenv("COMET_PROJECT_NAME")

# --------------------------------------
# ✅ Initialize Clients
# --------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------
# ✅ Streamlit UI Setup
# --------------------------------------
st.set_page_config(page_title="Upload and Embed Document into RAG System", page_icon="📄")
st.title("📄 Upload and Embed Document into RAG System")

uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
chunk_size = st.slider("Chunk size (in characters)", 300, 1000, 500, 100)

# --------------------------------------
# ✅ Helper: Extract Text from PDF
# --------------------------------------
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --------------------------------------
# ✅ Process Upload
# --------------------------------------
if uploaded_file:
    filename = uploaded_file.name
    filetype = filename.split(".")[-1].lower()

    # ✅ Extract text content
    if filetype == "txt":
        file_text = uploaded_file.read().decode("utf-8")
    elif filetype == "pdf":
        file_text = extract_text_from_pdf(uploaded_file.read())
    else:
        st.error("❌ Unsupported file type.")
        st.stop()

    # ✅ Chunk the text
    chunks = [file_text[i:i + chunk_size] for i in range(0, len(file_text), chunk_size)]
    st.write(f"📦 Found {len(chunks)} chunks to embed and store.")

    # ✅ Upload and embed
    if st.button("Upload to Supabase"):
        with st.spinner("🔄 Embedding and uploading..."):
            progress = st.progress(0)

            # ✅ Comet Logging
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
                    # ✅ Get embedding
                    embedding = client.embeddings.create(
                        input=chunk,
                        model="text-embedding-ada-002"
                    ).data[0].embedding

                    # ✅ Insert into Supabase
                    response = supabase.table("papers").insert({
                        "title": f"Upload: {filename}",
                        "chunk": chunk,
                        "embedding": embedding
                    }).execute()

                    # ✅ Debug log (display any errors)
                    st.write(f"✅ Chunk {i+1} inserted", response)

                except Exception as e:
                    st.error(f"❌ Failed to insert chunk {i+1}: {e}")

                progress.progress((i + 1) / len(chunks))

            experiment.end()
            st.success(f"✅ Uploaded {len(chunks)} chunks to Supabase and logged to Comet!")
