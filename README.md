# RAG Chat App

This project is a locally hosted Retrieval-Augmented Generation (RAG) system integrated into a Streamlit chat application. It loads multiple PDF documents, splits them into chunks, stores the chunks in a Qdrant vector database, and uses a RetrievalQA chain (powered by ChatGPT) to answer questions about the content.

## Features

- Load and split multiple PDFs into manageable chunks.
- Store documents chunks in Qdrant using vector embeddings.
- Use a RetrievalQA chain with ChatGPT (GPT-4o-mini) to answer questions.
- Chat interface built with Streamlit using `st.chat_message` for a modern UI.

## Requirements

- Python 3.11 or later
- [Docker](https://docs.docker.com/get-docker/) (for running Qdrant)
- Required Python packages:
  - streamlit
  - langchain-community
  - langchain-openai
  - qdrant-client
  - pypdf2

Install the Python dependencies using Poetry or pip. For example, using Poetry:

```bash
poetry install
```

## Setup

### Qdrant Setup:
Make sure Qdrant is running. You can start it using Docker:
```bash
sudo docker run -d -p 6333:6333 qdrant/qdrant:latest
```

### PDF Documents:
Place your PDF documents in the project folder and update the PDF_PATH in streamlit_app.py accordingly.

### Run the App:
Start the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

### Chat Interface:
Open your browser to the provided URL (e.g., http://localhost:8501) and start asking questions about the PDF content.