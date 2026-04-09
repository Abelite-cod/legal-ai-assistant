# Gemini PDF RAG System

This project is a Retrieval-Augmented Generation (RAG) system built with Python, LangChain, and Google's Gemini model.

The goal of the project is to allow users ask questions about a PDF document and receive accurate answers based strictly on the content of that document.

Instead of the AI making things up, the system retrieves relevant sections from the document and uses them as context for the answer.

I built this to explore how modern AI assistants work internally and to learn how production-level RAG pipelines are designed.

---

## How it works

The pipeline follows these steps:

1. Load a PDF document
2. Split the document into smaller chunks
3. Convert each chunk into embeddings
4. Store the embeddings inside a Chroma vector database
5. Retrieve relevant chunks when a user asks a question
6. Send the retrieved context to Gemini to generate the final answer

This ensures answers stay grounded in the source document.

---

## Tech Stack

- Python
- LangChain
- Google Gemini
- HuggingFace Embeddings
- Chroma Vector Database

---

## Project Structure

project/
│
├── rag_system.py
├── requirements.txt
├── README.md
├── .env
├── chroma_db/
└── ITI_AgenticAI_Final.pdf

---

## Installation

Clone the repository:

git clone https://github.com/Aelitecod/gemini-pdf-rag-system.git

cd gemini-pdf-rag-system

Install dependencies:

pip install -r requirements.txt

---

## Environment Variables

Create a `.env` file and add your Google API key.

GOOGLE_API_KEY=your_api_key_here

You can get an API key from Google AI Studio.

---

## Running the Project

Run the script:

python rag_system.py

You will then be able to ask questions about the PDF in the terminal.

Example:

Your question: What is Agentic AI?

The system will retrieve relevant sections from the document and generate an answer using Gemini.

---

## Features

- PDF document question answering
- Retrieval-Augmented Generation pipeline
- Query rewriting for better search
- Context-limited prompting to avoid hallucinations
- Source page references

---

## Example Workflow

User question:

What is agentic AI architecture?

Pipeline:

User Question
↓
Query Rewriting
↓
Vector Search
↓
Context Builder
↓
Gemini LLM
↓
Answer + Sources

---

## Why I Built This

I wanted to understand how modern AI assistants like ChatGPT use retrieval systems to ground their responses in real data.

This project helped me learn about vector databases, embeddings, prompt design, and building AI pipelines with LangChain.

---

## Future Improvements

Possible improvements include:

- Hybrid search (vector + keyword search)
- Reranking models for better retrieval
- FastAPI backend for deployment
- Streaming responses
- Multi-document support

---

## Author

Abel Okagbare

Computer Science graduate interested in AI systems, backend engineering, and building machine learning applications.
