"""
Utility module for RAG system initialization.

This module contains functions for:
- Loading and splitting a PDF into document chunks.
- Creating a Qdrant vector store.
- Building a RetrievalQA chain using a custom prompt pulled from LangChain Hub.
- Initializing the RAG system from a file or from all PDFs in a directory.

Ensure that you set the LANGSMITH_API_KEY environment variable to pull the prompt.
"""

import os
from typing import Any, List

from langchain import hub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def load_and_split_pdf(
    pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Any]:
    """
    Load a PDF file and split its text into chunks.

    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List: List of document chunks.
    """
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def load_all_pdfs_from_directory(
    pdf_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Any]:
    """
    Load all PDF files from a directory and combine their document chunks.

    Args:
        pdf_dir (str): Path to the directory containing PDF files.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.

    Returns:
        List: Combined list of document chunks from all PDFs.

    Raises:
        NotADirectoryError: If pdf_dir is not a valid directory.
    """
    if not os.path.isdir(pdf_dir):
        raise NotADirectoryError(f"{pdf_dir} is not a valid directory")
    all_docs = []
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            full_path = os.path.join(pdf_dir, filename)
            docs = load_and_split_pdf(full_path, chunk_size, chunk_overlap)
            all_docs.extend(docs)
    return all_docs


def create_qdrant_vectorstore(
    docs: List[Any],
    collection_name: str = "pdf_docs",
    host: str = "localhost",
    port: int = 6333,
) -> Qdrant:
    """
    Create a Qdrant vector store from the provided document chunks.

    Args:
        docs (List): List of document chunks.
        collection_name (str): Name of the Qdrant collection.
        host (str): Hostname where Qdrant is running.
        port (int): Port number for Qdrant.

    Returns:
        Qdrant: An instance of the Qdrant vector store.
    """
    embeddings = OpenAIEmbeddings()
    qdrant_url = f"http://{host}:{port}"
    vectorstore = Qdrant.from_documents(
        docs, embeddings, collection_name=collection_name, url=qdrant_url
    )
    return vectorstore


def create_retrieval_qa_chain(vectorstore: Qdrant) -> RetrievalQA:
    """
    Create a RetrievalQA chain using ChatGPT and a custom prompt pulled from LangChain Hub.

    The prompt template is pulled from hub.pull("daethyra/rag-prompt"). If the returned
    object is a ChatPromptTemplate (i.e. has a 'template' attribute), then its underlying
    string is used.

    Returns:
        RetrievalQA: A RetrievalQA chain instance.
    """
    rag_prompt = hub.pull("daethyra/rag-prompt").messages[0].prompt.template
    prompt = PromptTemplate(
        template=rag_prompt, input_variables=["context", "question"]
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=1.0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def init_rag_system(pdf_path: str) -> RetrievalQA:
    """
    Initialize the RAG system. If pdf_path is a directory, load all PDFs from it;
    if it is a file, load that single file.

    Args:
        pdf_path (str): Path to a PDF file or a directory containing PDFs.

    Returns:
        RetrievalQA: The initialized RetrievalQA chain.

    Raises:
        FileNotFoundError: If pdf_path is not found.
    """
    if os.path.isdir(pdf_path):
        docs = load_all_pdfs_from_directory(pdf_path)
    elif os.path.isfile(pdf_path):
        docs = load_and_split_pdf(pdf_path)
    else:
        raise FileNotFoundError(f"PDF path '{pdf_path}' not found.")
    vectorstore = create_qdrant_vectorstore(docs)
    qa_chain = create_retrieval_qa_chain(vectorstore)
    return qa_chain
