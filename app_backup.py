import streamlit as st
import requests

st.set_page_config(page_title="RAG Assistant", page_icon="💬")

# ----------------------------------------
# UI
# ----------------------------------------
st.title("💬 Ask a Question About Stored Knowledge")
st.markdown("Powered by FastAPI, OpenAI, PostgreSQL + pgvector")

# Question input
user_question = st.text_input("Enter your question")

# Button
if st.button("Generate Answer") and user_question:
    with st.spinner("Thinking..."):

        # Send POST request to FastAPI
        response = requests.post(
            "http://localhost:8000/generate",
            json={"question": user_question}
        )

        if response.status_code == 200:
            data = response.json()
            st.success("Answer:")
            st.write(data["answer"])
        else:
            st.error("Error from API. Please check FastAPI is running.")
