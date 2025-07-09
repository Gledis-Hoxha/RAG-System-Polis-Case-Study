# AI Assistant for School Website

This project is an AI-powered assistant built with Streamlit, designed to help students and new enrollees easily find relevant information from the school website by uploading PDF documents and asking questions in natural language. It implements a Retrieval-Augmented Generation (RAG) system to process documents and return accurate, context-aware answers.

## Features

- Upload and process PDF documents (first 10 pages)
- Automatic text cleaning and chunking
- Embedding generation and similarity search
- Question-answering interface with context retrieval
- Admin options for reusing or caching processed files

## Project Workflow

Below is a high-level overview of how the system works:

1. **User uploads a PDF** via the Streamlit interface  
2. **Text is extracted** from the first 10 pages  
3. The content is **cleaned and chunked**  
4. The system checks if the file was already processed  
   - If **not**, it generates embeddings for each chunk  
   - If **yes**, it loads embeddings from a cache (session or JSON)  
5. The user can **ask a question**  
6. The question is embedded and **compared with the document chunks**  
7. The **most similar chunks** are selected  
8. A prompt is composed using context + question and passed to the AI model for answering

## Technologies Used

- Python  
- Streamlit  
- OpenAI or LLM API (for answer generation)  
- Sentence transformers or similar (for embeddings)  
- JSON/session state for caching

## How to Run

```bash
# Clone the project
git clone https://github.com/yourusername/yourproject.git
cd yourproject

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
