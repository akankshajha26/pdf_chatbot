# GenAI PDF Chatbot

📌 ##Description

GenAI PDF Chatbot is an interactive chatbot built to query information directly from multiple PDF documents. Using OpenAI embeddings and FAISS vector storage, the bot can answer domain-specific questions with high accuracy. The user interface is designed with Streamlit, enabling smooth navigation and real-time querying.

✨ ##Features

Interactive chatbot UI built with Streamlit.

Retrieval-Augmented Generation (RAG) for context-aware responses.

Multi-PDF querying support, fetching context from multiple documents simultaneously.

Seamless querying from PDF content using FAISS vector search.

Strict alignment with PDF data, avoiding unwanted summarization through prompt engineering.

Displays chat history for reference during conversations.

🔧 ##Tech Stack

Backend: Python, LangChain, OpenAI

Frontend: Streamlit

Storage: FAISS (Facebook AI Similarity Search)

🚀 ##Usage

Upload the PDF documents you want to query.

Ask questions directly to the chatbot through the Streamlit UI.

The bot fetches relevant chunks from the PDFs and responds accurately.

🚀 ##Future Enhancements

Integrate memory-based conversational capabilities.

Expand multi-PDF querying to support cross-document reasoning.

