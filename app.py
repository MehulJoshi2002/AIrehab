import streamlit as st
import os
import sys
import tempfile
from dotenv import load_dotenv

# Load env first
load_dotenv()

try:
    from groq import Groq
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    IMPORTS_SUCCESS = True
    error_message = None
    
except Exception as e:
     IMPORTS_SUCCESS = False
     error_message = f"Import error: {e}"

# Initialize Groq
groq_api_key = os.getenv("GROQ_API_KEY") 

class RehabAssistant:
    def __init__(self):
        self.vector_store = None
        if IMPORTS_SUCCESS:
            self.groq_client = Groq(api_key=groq_api_key)
        else:
            self.groq_client = None

    def load_pdf(self, pdf_file):
        if not IMPORTS_SUCCESS:
            return False, f"Imports failed: {error_message}"
            
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                path = tmp.name

            loader = PyPDFLoader(path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            self.vector_store = Chroma.from_documents(chunks, embeddings)

            os.unlink(path)
            return True, f"Processed {len(docs)} pages, {len(chunks)} chunks created."

        except Exception as e:
            return False, f"Error loading PDF: {e}"

    def ask(self, question, patient_context=""):
        if not IMPORTS_SUCCESS:
            return f"Imports failed: {error_message}"
            
        if self.vector_store is None:
            return "❗ Upload a PDF first."

        try:
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
You are a medical physiotherapy assistant. Use the context below to answer the patient's question professionally.

PATIENT CONTEXT:
{patient_context}

PDF EXERCISE CONTEXT:
{context}

QUESTION:
{question}

Provide a medically accurate, helpful physiotherapy answer.
"""

            res = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a professional rehab assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.25,
                max_tokens=800
            )

            answer = res.choices[0].message.content

            src = "\n".join([
                f"- Page {d.metadata.get('page','?')} → {d.page_content[:140]}..."
                for d in docs
            ])

            return f"### Answer:\n{answer}\n\n---\n### Sources:\n{src}"

        except Exception as e:
            return f"Error while answering: {e}"


# ---------------------
# STREAMLIT UI
# --------------------
def main():
    st.set_page_config(page_title="AI Rehab Assistant", page_icon="🏥", layout="wide")

    st.title("🏥 AI Rehab Assistant")
    st.write("Upload physiotherapy PDFs and ask rehab-related questions.")
    
    # Debug info
    with st.expander("Debug Info"):
        st.write(f"Imports successful: {IMPORTS_SUCCESS}")
        if error_message:
            st.error(f"Import error: {error_message}")
        st.write(f"Groq API key: {'Set' if groq_api_key else 'Missing'}")
        st.write(f"Python executable: {sys.executable}")

    if not IMPORTS_SUCCESS:
        st.error("❌ Required packages not installed properly.")
        st.code("pip install groq langchain-community langchain-text-splitters chromadb sentence-transformers")
        return

    # init assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = RehabAssistant()

    # sidebar upload
    with st.sidebar:
        st.header("📁 Upload Exercise PDF")
        pdf = st.file_uploader("Upload PDF", type="pdf")

        if pdf is not None:
            success, msg = st.session_state.assistant.load_pdf(pdf)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.header("👤 Patient Info")
        context = st.text_area("Optional patient details:")

    st.header("💬 Ask a physiotherapy question")
    question = st.text_input("Ask something about the rehab exercises:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Thinking..."):
                response = st.session_state.assistant.ask(question, context)
                st.markdown(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
