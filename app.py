# --------------------------------------
# ✅ Imports
# --------------------------------------
import streamlit as st
import requests
import os
from dotenv import load_dotenv

# --------------------------------------
# ✅ Load environment variables
# --------------------------------------
load_dotenv()

# --------------------------------------
# ✅ Connect to FastAPI Backend on Render
# --------------------------------------
FASTAPI_URL = "https://rag-uploader.onrender.com/generate"  # ⬅️ Replace localhost with your Render URL

# --------------------------------------
# ✅ Streamlit Page Config
# --------------------------------------
st.set_page_config(page_title="📚 Ask a Question About Stored Knowledge")
st.title("💬 Ask a Question About Stored Knowledge")
st.markdown("Powered by FastAPI, OpenAI, PostgreSQL + pgvector")

# --------------------------------------
# ✅ Question Input Box
# --------------------------------------
question = st.text_input("Enter your question")

# --------------------------------------
# ✅ Generate Answer Button
# --------------------------------------
if st.button("Generate Answer"):

    # Show spinner while processing
    with st.spinner("🧠 Generating answer..."):

        # Send POST request to FastAPI with your question
        try:
            response = requests.post(FASTAPI_URL, json={"question": question})
            result = response.json()

            # Display answer
            st.success("Answer:")
            st.write(result.get("answer", "❌ No answer found."))

        except Exception as e:
            st.error(f"Request failed: {e}")
