import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load .env file (locally). On Streamlit Cloud, you enter secrets manually.
load_dotenv()

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ---------------------------
# REHAB ASSISTANT CLASS
# ---------------------------
class RehabAssistant:
    def __init__(self):
        self.vector_store = None
        api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_client = Groq(api_key=api_key)

    def load_pdf(self, pdf_file):
        """Load and process PDF into embeddings"""
        try:
            # Save temp PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                path = tmp.name

            # Load PDF
            loader = PyPDFLoader(path)
            docs = loader.load()

            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(docs)

            # Vector store
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.vector_store = Chroma.from_documents(chunks, embeddings)

            os.unlink(path)
            return True, f"Processed {len(docs)} pages and {len(chunks)} text chunks."

        except Exception as e:
            return False, f"Error loading PDF: {e}"

    def ask(self, question, patient_context=""):
        """Answer questions using embeddings + Groq"""
        if not self.vector_store:
            return "⚠ Upload a PDF first."

        try:
            # Search most relevant content
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            # Build LLM prompt
            prompt = f"""
You are an expert physiotherapy assistant. Use the PDF content below to answer.

PATIENT CONTEXT:
{patient_context}

PDF EXERCISE CONTENT:
{context}

QUESTION:
{question}

Provide a clear, accurate medical physiotherapy explanation.
"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a professional physiotherapy assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.25
            )

            answer = response.choices[0].message.content

            # Add sources
            sources = "\n".join(
                f"- Page {d.metadata.get('page', '?')} → {d.page_content[:140]}..."
                for d in docs
            )

            return f"### Answer\n{answer}\n\n---\n### Sources\n{sources}"

        except Exception as e:
            return f"Error answering question: {e}"


# ---------------------------
# STREAMLIT UI
# ---------------------------
def main():
    st.set_page_config(page_title="AI Rehab Assistant", page_icon="🏥", layout="wide")

    st.title("🏥 AI Rehab Assistant")
    st.write("Upload physiotherapy PDFs and ask rehabilitation-related questions.")

    # Initialize assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = RehabAssistant()

    # Sidebar – PDF Upload
    with st.sidebar:
        st.header("📁 Upload Exercise PDF")
        pdf = st.file_uploader("Upload a physiotherapy PDF", type="pdf")

        if pdf is not None:
            success, msg = st.session_state.assistant.load_pdf(pdf)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.header("👤 Patient Context (Optional)")
        patient_context = st.text_area("Describe the patient:", height=120)

    # Main question input
    st.header("💬 Ask a physiotherapy question")
    question = st.text_input("Type your question:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Analyzing and preparing response..."):
                reply = st.session_state.assistant.ask(question, patient_context)
                st.markdown(reply)
        else:
            st.warning("Please enter a question.")


if __name__ == "__main__":
    main()
