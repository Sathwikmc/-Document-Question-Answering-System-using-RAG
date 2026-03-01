# 💊 Paracetamol Document QA System (RAG)

A Retrieval-Augmented Generation (RAG) application that allows users to upload Paracetamol-related PDF documents and ask intelligent, context-aware questions. Built with **Streamlit**, **LangChain**, and **Groq** (100% free inference).

---

## 🚀 Features

- 📄 **PDF Upload & Processing** — Upload any Paracetamol-related research PDF
- 🔍 **Semantic Search** — FAISS vector store with HuggingFace sentence embeddings
- 🤖 **Multiple LLM Providers** — Groq (free), OpenAI (paid), HuggingFace (free)
- 💬 **Context-Grounded Answers** — Answers sourced directly from document content
- 📚 **Source Transparency** — View the exact document chunks used per answer

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Streamlit |
| LLM Orchestration | LangChain |
| LLM (Recommended) | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| PDF Loader | PyPDF |

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/paracetamol-rag.git
cd paracetamol-rag
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
```bash
cp .env.example .env
# Edit .env and add your API key(s)
```

Get a **FREE** Groq API key at [console.groq.com](https://console.groq.com) — no credit card required.

### 5. Run the App
```bash
streamlit run app.py
```

---

## 🔑 API Key Setup

| Provider | Cost | URL |
|----------|------|-----|
| Groq ✅ Recommended | **FREE** | https://console.groq.com |
| OpenAI | Paid | https://platform.openai.com/api-keys |
| HuggingFace | Free (limited) | https://huggingface.co/settings/tokens |

---

## 📁 Project Structure

```
paracetamol-rag/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

---

## 🧠 How It Works

```
User uploads PDF
        ↓
PDF is split into chunks (1000 tokens, 200 overlap)
        ↓
Chunks are embedded using HuggingFace sentence-transformers
        ↓
Embeddings stored in FAISS vector database
        ↓
User asks a question
        ↓
Question is embedded → Top-3 similar chunks retrieved
        ↓
Chunks + question sent to Groq LLM
        ↓
Answer generated and displayed with source chunks
```

---

## ⚙️ Configuration

Edit the `CONFIG` dictionary in `app.py` to customize behaviour:

```python
CONFIG = {
    "chunk_size": 1000,           # Tokens per chunk
    "chunk_overlap": 200,         # Overlap between chunks
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_provider": "groq",       # groq | openai | huggingface
    "llm_model": "llama-3.3-70b-versatile",
    "temperature": 0.5,           # 0 = deterministic, 1 = creative
    "max_tokens": 512,
    "retrieval_k": 3,             # Number of chunks to retrieve
}
```

---

## 📋 Supported Groq Models (All FREE)

| Model | Context | Speed |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | 128K | Fast |
| `llama-3.1-8b-instant` | 128K | Fastest |
| `llama3-70b-8192` | 8K | Fast |
| `llama3-8b-8192` | 8K | Fastest |
| `gemma2-9b-it` | 8K | Fast |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m 'Add your feature'`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is for educational purposes. See LICENSE for details.

---

## 👤 Author

Built as part of an NLP/RAG coursework submission.
