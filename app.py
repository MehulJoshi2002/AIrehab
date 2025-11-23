import streamlit as st
import os
import tempfile
import numpy as np
from dotenv import load_dotenv
import pypdf
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
        self.groq_client = None
        self.setup_groq_client()
    
    def setup_groq_client(self):
        """Setup Groq client with proper error handling"""
        try:
            api_key = os.getenv("GROQ_API_KEY")
            
            if not api_key:
                st.error("❌ GROQ_API_KEY environment variable is empty")
                return
            
            # Validate API key format
            if not api_key.startswith('gsk_'):
                st.error("❌ API key format invalid - should start with 'gsk_'")
                return
            
            # Import Groq only when needed
            from groq import Groq
            self.groq_client = Groq(api_key=api_key)
            st.success("✅ Groq client initialized successfully")
            
        except ImportError:
            st.error("❌ Groq package not installed. Add 'groq' to requirements.txt")
        except Exception as e:
            st.error(f"❌ Failed to initialize Groq client: {str(e)}")

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
            return "⚠ Groq client not properly initialized. Check your API key."

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

Please provide a helpful, professional response based on the PDF content.

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
    
    # Debug API key
    api_key = os.getenv("GROQ_API_KEY")
    with st.expander("🔧 Debug API Key Status"):
        if api_key:
            st.success(f"✅ API Key Found: {api_key[:10]}...{api_key[-10:]}")
            if api_key.startswith('gsk_'):
                st.success("✅ API Key format looks correct")
            else:
                st.error("❌ API Key should start with 'gsk_'")
        else:
            st.error("❌ No API Key found in environment variables")

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
        if st.session_state.assistant.groq_client:
            st.success("✅ Groq connection ready")
        else:
            st.error("❌ Groq connection failed")

if __name__ == "__main__":
    main()
