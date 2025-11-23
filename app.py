import streamlit as st
import os
import tempfile
import pypdf
from groq import Groq

# Initialize Groq client with error handling
@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables")
        return None
    
    # Clean the API key
    api_key = api_key.strip()
    
    if not api_key.startswith('gsk_'):
        st.error("❌ Invalid API key format - should start with 'gsk_'")
        return None
        
    try:
        client = Groq(api_key=api_key)
        # Test the connection
        test_response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5
        )
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Groq: {str(e)}")
        return None

class SimpleRehabAssistant:
    def __init__(self):
        self.client = get_groq_client()
        self.documents = []
    
    def load_pdf(self, pdf_file):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            pdf_reader = pypdf.PdfReader(tmp_path)
            self.documents = []
            
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text.strip():
                    self.documents.append({
                        'content': text,
                        'page': page_num + 1
                    })
            
            os.unlink(tmp_path)
            return True, f"Loaded {len(self.documents)} pages"
            
        except Exception as e:
            return False, f"PDF error: {str(e)}"
    
    def ask_question(self, question, patient_context=""):
        if not self.documents:
            return "Please upload a PDF first."
        
        if not self.client:
            return "Groq client not available. Check API key."
        
        try:
            # Use the first few pages as context (simplified approach)
            context = "\n\n".join([
                f"Page {doc['page']}: {doc['content'][:500]}"
                for doc in self.documents[:2]  # Use first 2 pages
            ])
            
            prompt = f"""
Patient Context: {patient_context}

PDF Content:
{context}

Question: {question}

Please provide a helpful physiotherapy answer based on the PDF content above.
"""
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are a professional physiotherapy assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return f"**Answer:** {response.choices[0].message.content}"
            
        except Exception as e:
            return f"Error: {str(e)}"

def main():
    st.set_page_config(page_title="AI Rehab Assistant", page_icon="🏥")
    
    st.title("🏥 AI Rehab Assistant")
    
    # Debug info
    with st.expander("🔧 Connection Status"):
        api_key = os.getenv("GROQ_API_KEY")
        if api_key:
            st.success(f"✅ API Key Found: {api_key[:8]}...{api_key[-8:]}")
        else:
            st.error("❌ No API Key in environment")
    
    # Initialize assistant
    if "assistant" not in st.session_state:
        st.session_state.assistant = SimpleRehabAssistant()
    
    # PDF Upload
    st.subheader("📁 Upload Exercise PDF")
    pdf_file = st.file_uploader("Choose PDF", type="pdf")
    
    if pdf_file and st.button("Process PDF"):
        success, message = st.session_state.assistant.load_pdf(pdf_file)
        if success:
            st.success(message)
        else:
            st.error(message)
    
    # Patient Context
    st.subheader("👤 Patient Information")
    patient_context = st.text_area("Condition/context (optional):")
    
    # Question
    st.subheader("💬 Ask a Question")
    question = st.text_input("Enter your question:")
    
    if st.button("Get Answer") and question:
        if st.session_state.assistant.documents:
            with st.spinner("Generating response..."):
                answer = st.session_state.assistant.ask_question(question, patient_context)
                st.markdown(answer)
        else:
            st.warning("Please upload and process a PDF first.")

if __name__ == "__main__":
    main()
