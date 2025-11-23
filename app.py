import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import with error handling
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from groq import Groq
    IMPORTS_SUCCESS = True
except ImportError as e:
    st.error(f"Import error: {e}")
    IMPORTS_SUCCESS = False

# ----------------------------
# REHAB ASSISTANT CLASS
# ----------------------------
class RehabAssistant:
    def __init__(self):
        self.vector_store = None
        if IMPORTS_SUCCESS:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                st.error("❌ GROQ_API_KEY not found in environment variables")
            self.groq_client = Groq(api_key=api_key) if api_key else None
        else:
            self.groq_client = None

    def load_pdf(self, pdf_file):
        """Load PDF and create embeddings."""
        if not IMPORTS_SUCCESS:
            return False, "Required imports failed"
            
        if not self.groq_client:
            return False, "Groq client not initialized - check API key"

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
            return True, f"Loaded {len(docs)} pages and created {len(chunks)} chunks."

        except Exception as e:
            return False, f"PDF error: {e}"

    def ask(self, question, patient_context=""):
        if not IMPORTS_SUCCESS:
            return "Required imports failed - check dependencies"
            
        if self.vector_store is None:
            return "⚠ Please upload a PDF first."

        if not self.groq_client:
            return "⚠ Groq API key not configured"

        try:
            # Retrieve relevant chunks
            docs = self.vector_store.similarity_search(question, k=3)
            context = "\n\n".join([d.page_content for d in docs])

            prompt = f"""
You are an expert physiotherapy assistant.
Use ONLY the PDF content below to answer.

PATIENT CONTEXT:
{patient_context}

PDF CONTEXT:
{context}

QUESTION: {question}

Provide a medically accurate, clear, helpful answer.
"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a physiotherapy specialist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=700,
                temperature=0.25,
            )

            answer = response.choices[0].message.content

            # Sources
            sources = "\n".join(
                f"- Page {d.metadata.get('page','?')} → {d.page_content[:120]}..."
                for d in docs
            )

            return f"### Answer\n{answer}\n\n---\n### Sources\n{sources}"

        except Exception as e:
            return f"LLM error: {e}"


# ----------------------------------------
# STREAMLIT UI
# ----------------------------------------
def main():
    st.set_page_config(page_title="AI Rehab Assistant", page_icon="🏥", layout="wide")

    st.title("🏥 AI Rehab Assistant")
    st.write("Upload a physiotherapy PDF and ask rehab-related questions.")

    if not IMPORTS_SUCCESS:
        st.error("""
        ❌ Some required packages failed to load. This is usually due to:
        - Python 3.13 compatibility issues
        - Missing dependencies in requirements.txt
        - Version conflicts
        
        Please check the Streamlit Cloud logs for detailed error messages.
        """)
        return

    # Initialize assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = RehabAssistant()

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("""
        ❌ GROQ_API_KEY not found!
        
        On Streamlit Cloud, add your Groq API key in the Secrets section:
        1. Go to your app dashboard
        2. Click 'Settings' → 'Secrets'
        3. Add: GROQ_API_KEY=your_actual_api_key_here
        """)

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Physiotherapy PDF")
        pdf = st.file_uploader("Choose PDF file", type="pdf")

        if pdf:
            success, msg = st.session_state.assistant.load_pdf(pdf)
            if success:
                st.success(msg)
            else:
                st.error(msg)

        st.header("👤 Patient Information")
        patient_context = st.text_area("Describe patient condition (optional):")

    # Main Question Area
    st.header("💬 Ask a rehabilitation question")
    question = st.text_input("Type your question here:")

    if st.button("Get Answer"):
        if question.strip():
            with st.spinner("Analyzing PDF and generating response..."):
                result = st.session_state.assistant.ask(question, patient_context)
                st.markdown(result)
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()
