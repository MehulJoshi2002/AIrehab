import streamlit as st
import os
import tempfile
import numpy as np
from dotenv import load_dotenv
from groq import Groq
import pypdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load environment variables
load_dotenv()

# Initialize components
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class RehabAssistant:
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.embedding_model = load_embedding_model()
        api_key = os.getenv("GROQ_API_KEY")
        self.groq_client = Groq(api_key=api_key) if api_key else None

    def load_pdf(self, pdf_file):
        """Load PDF and extract text without LangChain"""
        try:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            # Extract text from PDF using pypdf
            pdf_reader = pypdf.PdfReader(tmp_path)
            documents = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    documents.append({
                        'content': text,
                        'page': page_num + 1,
                        'metadata': {'page': page_num + 1}
                    })
            
            # Create embeddings
            texts = [doc['content'] for doc in documents]
            embeddings = self.embedding_model.encode(texts)
            
            self.documents = documents
            self.embeddings = embeddings
            
            os.unlink(tmp_path)
            return True, f"Loaded {len(documents)} pages from PDF."
            
        except Exception as e:
            return False, f"Error processing PDF: {str(e)}"

    def find_relevant_documents(self, query, k=3):
        """Find most relevant documents using cosine similarity"""
        if not self.documents or self.embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top k documents
        top_indices = np.argsort(similarities)[-k:][::-1]
        relevant_docs = [self.documents[i] for i in top_indices]
        
        return relevant_docs

    def ask(self, question, patient_context=""):
        if not self.documents:
            return "⚠ Please upload a PDF first."
        
        if not self.groq_client:
            return "⚠ Groq API key not configured"

        try:
            # Find relevant documents
            relevant_docs = self.find_relevant_documents(question, k=3)
            
            if not relevant_docs:
                return "No relevant information found in the PDF for your question."
            
            # Build context from relevant documents
            context = "\n\n".join([
                f"Page {doc['page']}:\n{doc['content'][:1000]}..." 
                for doc in relevant_docs
            ])

            prompt = f"""
You are an expert AI physiotherapy assistant. Use the information from the uploaded PDF to answer the patient's question.

PATIENT CONTEXT:
{patient_context}

RELEVANT INFORMATION FROM PDF:
{context}

PATIENT QUESTION:
{question}

Please provide a helpful, professional response based on the PDF content. If the information doesn't fully cover the question, provide the best guidance you can based on the available information.

ANSWER:
"""

            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a professional physiotherapy assistant. Provide accurate, helpful guidance based on the provided PDF content."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=1024,
                temperature=0.3,
            )

            answer = response.choices[0].message.content

            # Format response with sources
            sources_text = "\n".join([
                f"- Page {doc['page']}: {doc['content'][:150]}..."
                for doc in relevant_docs
            ])

            return f"""
### 💡 Answer

{answer}

---

### 📚 Sources from PDF

{sources_text}
"""

        except Exception as e:
            return f"Error generating response: {str(e)}"

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Rehab Assistant",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 AI Rehab Assistant")
    st.markdown("Upload physiotherapy exercise PDFs and get personalized guidance")
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("""
        ❌ GROQ_API_KEY not found!
        
        On Streamlit Cloud:
        1. Go to your app dashboard
        2. Click 'Settings' → 'Secrets'
        3. Add: GROQ_API_KEY=your_actual_api_key_here
        
        For local development, create a .env file with GROQ_API_KEY=your_key
        """)
        return

    # Initialize assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = RehabAssistant()

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Exercise PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload your physiotherapy exercise PDF"
        )
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    success, message = st.session_state.assistant.load_pdf(uploaded_file)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

        st.header("👤 Patient Information")
        patient_context = st.text_area(
            "Patient context (optional):",
            placeholder="e.g., 65-year-old with knee arthritis, post-surgery recovery...",
            height=100
        )

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        if not st.session_state.assistant.documents:
            st.info("👆 Please upload and process a PDF file to get started.")
        else:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What exercises are safe for knee pain? How many repetitions should I do?"
            )
            
            if st.button("Get Answer", type="primary") and question:
                with st.spinner("Analyzing PDF and generating response..."):
                    answer = st.session_state.assistant.ask(question, patient_context)
                    st.markdown(answer)
    
    with col2:
        st.header("📊 Quick Info")
        if st.session_state.assistant.documents:
            st.success(f"✅ PDF loaded: {len(st.session_state.assistant.documents)} pages")
            st.info("💡 Try asking about:")
            st.write("- Safe exercises for specific conditions")
            st.write("- Exercise repetitions and sets")
            st.write("- Safety precautions")
            st.write("- Recovery timelines")

if __name__ == "__main__":
    main()
