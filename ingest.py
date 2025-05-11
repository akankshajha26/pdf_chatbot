import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path='C:/Users/jhaak/Downloads/Resume Projects/genai_qa_bots/.env')

# Load API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI api key is missing. Please set it as an environment variable.")

# Load PDF documents
folder_path = "C:/Users/jhaak/Downloads/Resume Projects/genai_qa_bots/docs"
all_documents = []

for filename in os.listdir(folder_path):
    if filename.endswith(".pdf"):
        file_path = os.path.join(folder_path, filename)
        print(f"Loading {file_path}")
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            if documents:
                print(f"{filename} loaded successfully")
            else:
                print(f"Failed to load {filename}")
            all_documents.extend(documents)
        except Exception as e:
            print(f"Error loading {filename}: {e}")

# Simple Text Splitter (No Page Transition Logic)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=600,   # Slight overlap to handle context flow
    separators=["\n\n", "\n", ".", "!", "?"]  # Basic splitting
)

# Split the documents
chunks = text_splitter.split_documents(all_documents)

# Embedding the data
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Store in FAISS Vectorstore
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the vectorstore locally
vectorstore.save_local("vectorstores/decision_tree_index")

print("Ingestion complete. Vector store saved.")