"""
Paracetamol Document QA System - RAG Application (FULLY CORRECTED)
Uses Groq API (100% FREE) with HuggingFace embeddings
Built with Streamlit, LangChain, and Groq
"""

import streamlit as st
import os
import tempfile
import logging
from typing import List, Optional
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain.llms.base import LLM
from dotenv import load_dotenv

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================================
# CONFIGURATION & ENVIRONMENT SETUP
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Get API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Configuration constants
CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_provider": "groq",  # "groq", "openai", or "huggingface"
    "llm_model": "llama-3.3-70b-versatile",  # ✅ UPDATED - New Groq model
    "temperature": 0.5,
    "max_tokens": 512,
    "retrieval_k": 3,
}

# ============================================================================
# GROQ LLM WRAPPER (100% FREE)
# ============================================================================

class GroqLLM(LLM):
    """
    Custom LLM wrapper for Groq API (100% FREE).
    Uses Groq's inference API for fast, free responses.
    """
    
    api_key: str
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = 0.5
    max_tokens: int = 512
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, api_key: str, model_name: str = "llama-3.3-70b-versatile", 
                 temperature: float = 0.5, max_tokens: int = 512, **kwargs):
        """Initialize Groq LLM"""
        super().__init__(api_key=api_key, model_name=model_name, 
                        temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call Groq API"""
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content
        except ImportError:
            raise Exception("Groq library not installed. Run: pip install groq")
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

# ============================================================================
# OPENAI LLM WRAPPER (PAID - OPTIONAL)
# ============================================================================

class OpenAILLM(LLM):
    """
    Custom LLM wrapper for OpenAI API (PAID).
    Optional alternative if you prefer OpenAI.
    """
    
    api_key: str
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.5
    max_tokens: int = 512
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", 
                 temperature: float = 0.5, max_tokens: int = 512, **kwargs):
        """Initialize OpenAI LLM"""
        super().__init__(api_key=api_key, model_name=model_name, 
                        temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "openai"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call OpenAI API"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except ImportError:
            raise Exception("OpenAI library not installed. Run: pip install openai")
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

# ============================================================================
# HUGGINGFACE LLM WRAPPER (FREE - OPTIONAL)
# ============================================================================

class HuggingFaceInferenceLLM(LLM):
    """
    Custom LLM wrapper for HuggingFace Inference API.
    Free but less reliable (frequent outages).
    """
    
    model_id: str
    temperature: float = 0.5
    max_tokens: int = 512
    hf_token: str = ""
    
    class Config:
        """Pydantic config"""
        arbitrary_types_allowed = True
        extra = "allow"
    
    def __init__(self, model_id: str, hf_token: str, temperature: float = 0.5, 
                 max_tokens: int = 512, **kwargs):
        """Initialize HuggingFace LLM"""
        super().__init__(model_id=model_id, hf_token=hf_token, 
                        temperature=temperature, max_tokens=max_tokens, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_inference"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
        """Call HuggingFace Inference API"""
        try:
            import requests
            
            api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 410:
                raise Exception(f"Model '{self.model_id}' is no longer available on HuggingFace")
            if response.status_code == 429:
                raise Exception("Rate limited by HuggingFace")
            if response.status_code == 503:
                raise Exception("HuggingFace service unavailable")
            
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], dict) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
            elif isinstance(result, dict) and "generated_text" in result:
                return result["generated_text"].strip()
            
            return str(result)
        except Exception as e:
            raise Exception(f"HuggingFace API error: {str(e)}")

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Paracetamol Research Assistant",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {padding: 2rem;}
    .stExpander {background-color: #f0f2f6;}
    h1, h2, h3 {color: #1f77b4;}
    .success-box {background-color: #d4edda; padding: 10px; border-radius: 5px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💊 Paracetamol Document QA System (RAG)")
st.markdown(
    "**Upload a PDF about Paracetamol and ask intelligent questions based on its content.**\n\n"
    "Powered by LangChain, Groq (FREE), and Retrieval-Augmented Generation (RAG)"
)

# ============================================================================
# VALIDATION & CONFIGURATION
# ============================================================================

def validate_configuration():
    """Validate that required API keys are present"""
    provider = CONFIG["llm_provider"]
    
    if provider == "groq":
        if not GROQ_API_KEY:
            st.error("❌ Groq API Key Missing")
            st.info(
                "Get FREE Groq API key:\n\n"
                "1. Visit: https://console.groq.com\n"
                "2. Sign up (free, no payment)\n"
                "3. Copy your API key\n"
                "4. Add to .env: `GROQ_API_KEY=gsk-...`"
            )
            return False
    elif provider == "openai":
        if not OPENAI_API_KEY:
            st.error("❌ OpenAI API Key Missing")
            st.info(
                "Get OpenAI API key:\n\n"
                "1. Visit: https://platform.openai.com/api-keys\n"
                "2. Create account (requires credit card)\n"
                "3. Copy your API key\n"
                "4. Add to .env: `OPENAI_API_KEY=sk-...`"
            )
            return False
    elif provider == "huggingface":
        if not HF_TOKEN:
            st.error("❌ HuggingFace Token Missing")
            st.info(
                "Get HuggingFace token:\n\n"
                "1. Visit: https://huggingface.co/settings/tokens\n"
                "2. Create token (free)\n"
                "3. Copy your token\n"
                "4. Add to .env: `HF_TOKEN=hf_...`"
            )
            return False
    
    return True

# ============================================================================
# DOCUMENT PROCESSING FUNCTIONS
# ============================================================================

def process_document(uploaded_file) -> List[Document]:
    """Load and process a PDF document into chunks"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            raise ValueError("PDF loaded but contains no pages")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def create_vector_db(chunks: List[Document]) -> FAISS:
    """Create FAISS vector database from chunks"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=CONFIG["embedding_model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        vector_db = FAISS.from_documents(chunks, embeddings)
        return vector_db
    except Exception as e:
        raise Exception(f"Failed to create vector database: {str(e)}")


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents"""
    return "\n\n".join(doc.page_content for doc in docs)


def initialize_llm():
    """Initialize LLM based on configuration"""
    try:
        provider = CONFIG["llm_provider"]
        
        if provider == "groq":
            if not GROQ_API_KEY:
                raise Exception("GROQ_API_KEY not found in environment")
            llm = GroqLLM(
                api_key=GROQ_API_KEY,
                model_name=CONFIG["llm_model"],
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
            )
        elif provider == "openai":
            if not OPENAI_API_KEY:
                raise Exception("OPENAI_API_KEY not found in environment")
            llm = OpenAILLM(
                api_key=OPENAI_API_KEY,
                model_name=CONFIG["llm_model"],
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
            )
        else:  # huggingface
            if not HF_TOKEN:
                raise Exception("HF_TOKEN not found in environment")
            llm = HuggingFaceInferenceLLM(
                model_id=CONFIG["llm_model"],
                hf_token=HF_TOKEN,
                temperature=CONFIG["temperature"],
                max_tokens=CONFIG["max_tokens"],
            )
        return llm
    except Exception as e:
        raise Exception(f"Failed to initialize LLM: {str(e)}")

# ============================================================================
# MAIN APPLICATION UI
# ============================================================================

# Sidebar
with st.sidebar:
    st.header("📋 About")
    st.info(
        "This application uses **Retrieval-Augmented Generation (RAG)** to answer "
        "questions about Paracetamol documents. The system retrieves the most relevant "
        "document chunks and uses an LLM to generate accurate answers."
    )

    st.header("⚙️ System Configuration")
    st.json(CONFIG)
    
    st.header("🔑 API Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        if GROQ_API_KEY:
            st.success("✅ Groq API")
        else:
            st.warning("⚠️ Groq API")
    with col2:
        if OPENAI_API_KEY:
            st.success("✅ OpenAI")
        else:
            st.warning("⚠️ OpenAI")
    with col3:
        if HF_TOKEN:
            st.success("✅ HuggingFace")
        else:
            st.warning("⚠️ HuggingFace")
    
    # Model selector in sidebar
    st.header("🤖 Model Selection")
    available_groq_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "gemma2-9b-it",
        "compound-beta",
    ]
    selected_model = st.selectbox(
        "Select Groq Model:",
        available_groq_models,
        index=0,
        help="Choose the LLM model for generating answers"
    )
    CONFIG["llm_model"] = selected_model

# Validate configuration
if not validate_configuration():
    st.stop()

# Main content
uploaded_file = st.file_uploader(
    "📁 Upload Paracetamol PDF Document",
    type="pdf",
    help="Select a PDF file containing Paracetamol-related documentation",
)

if uploaded_file:
    # Process document
    if "vector_db" not in st.session_state or "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("🔄 Processing document and creating embeddings..."):
            try:
                chunks = process_document(uploaded_file)
                st.session_state.vector_db = create_vector_db(chunks)
                st.session_state.current_file = uploaded_file.name
                st.session_state.chunk_count = len(chunks)
                st.success(
                    f"✅ Document indexed successfully! **{len(chunks)} chunks** created"
                )
            except Exception as e:
                st.error(f"❌ Error processing document:\n{str(e)}")
                st.stop()
    else:
        st.success(f"✅ Document already indexed! **{st.session_state.chunk_count} chunks** available")

    # Query input
    query = st.text_input(
        "❓ Ask a question about Paracetamol:",
        placeholder="e.g., What are the side effects of Paracetamol?",
        help="Type your question based on the uploaded document",
    )

    if query:
        with st.spinner("🔍 Searching for answers..."):
            try:
                # Initialize LLM
                llm = initialize_llm()

                # Create retriever
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": CONFIG["retrieval_k"]}
                )

                # Prompt template
                prompt_template = """You are an expert assistant specialized in answering questions about Paracetamol.

Your task:
1. Use ONLY the provided context to answer the question
2. If the answer is not in the context, clearly state you don't know
3. Provide accurate, clear, and concise answers
4. Cite specific information from the context when relevant

Context from document:
{context}

Question: {question}

Answer:"""

                prompt = ChatPromptTemplate.from_template(prompt_template)

                # Build RAG chain
                chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm
                    | StrOutputParser()
                )

                # Generate answer
                answer = chain.invoke(query)
                source_docs = retriever.invoke(query)

                # Display results
                st.subheader("📝 Answer:")
                st.write(answer)

                st.subheader(f"📚 Source Chunks (Top {CONFIG['retrieval_k']} Matches):")
                for i, doc in enumerate(source_docs, 1):
                    page_num = doc.metadata.get("page", "Unknown")
                    with st.expander(f"Chunk {i} — Page {page_num}"):
                        st.write(doc.page_content)

                st.info(
                    f"ℹ️ Retrieved {len(source_docs)} most relevant document chunks using model: **{CONFIG['llm_model']}**"
                )

            except Exception as e:
                st.error(f"❌ Error processing query:\n{str(e)}")
                with st.expander("📋 Error Details"):
                    import traceback
                    st.code(traceback.format_exc(), language="python")

    # System architecture
    with st.expander("⚙️ System Architecture Details"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Document Chunking")
            st.write(f"- **Chunk Size:** {CONFIG['chunk_size']} tokens")
            st.write(f"- **Chunk Overlap:** {CONFIG['chunk_overlap']} tokens")
            st.write("- **Splitter:** RecursiveCharacterTextSplitter")

            st.subheader("Embeddings Model")
            st.write(f"- **Model:** {CONFIG['embedding_model']}")
            st.write("- **Vector Store:** FAISS")
            st.write("- **Normalized:** Yes")

        with col2:
            st.subheader("LLM Configuration")
            st.write(f"- **Provider:** {CONFIG['llm_provider'].upper()}")
            st.write(f"- **Model:** {CONFIG['llm_model']}")
            st.write(f"- **Temperature:** {CONFIG['temperature']}")
            st.write(f"- **Max Tokens:** {CONFIG['max_tokens']}")
            st.write(f"- **Retrieval K:** {CONFIG['retrieval_k']}")

        st.divider()

        st.subheader("📊 RAG Pipeline")
        st.write(
            """
            1. **Document Loading**: PDF is loaded and parsed
            2. **Chunking**: Document is split into overlapping chunks
            3. **Embedding**: Chunks are converted to embeddings using HuggingFace
            4. **Vector Storage**: Embeddings are stored in FAISS
            5. **Retrieval**: User query retrieves top-k similar chunks
            6. **LLM Generation**: Groq/OpenAI/HuggingFace generates answer
            7. **Output**: Answer and source chunks displayed
            """
        )

else:
    st.info("👆 **Upload a PDF file to get started**")
    st.markdown(
        """
        ### 🚀 Getting Started:
        1. Click the **Upload Paracetamol PDF** button above
        2. Select your PDF document
        3. Wait for document processing and indexing
        4. Ask questions about the document content
        5. Review the answers and source chunks
        
        ### 🔑 API Configuration (Choose ONE):
        - **Groq** (Recommended): FREE, fast - https://console.groq.com
        - **OpenAI**: PAID, reliable - https://platform.openai.com
        - **HuggingFace**: FREE, less reliable - https://huggingface.co
        
        ### 🤖 Available Groq Models (All FREE):
        | Model | Description |
        |-------|-------------|
        | `llama-3.3-70b-versatile` | Best quality, 128K context |
        | `llama-3.1-8b-instant` | Fastest responses |
        | `llama3-70b-8192` | High quality, 8K context |
        | `llama3-8b-8192` | Fast & lightweight |
        | `gemma2-9b-it` | Google's model |
        """
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>💊 Paracetamol Document QA System | Built with Streamlit, LangChain & Groq</p>
        <p>Using Model: <strong>{}</strong></p>
    </div>
    """.format(CONFIG["llm_model"]),
    unsafe_allow_html=True,
)