import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables
load_dotenv(dotenv_path="C:/Users/jhaak/Downloads/Resume Projects/genai_qa_bots/.env")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is missing. Please set it as an environment variable.")

# Initialize LLM and Embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load vectorstore
vectorstore = FAISS.load_local("vectorstores/decision_tree_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Custom prompt to enforce context and prevent summarization
template = """
You are an informative and reliable assistant. Your responsibility is to provide clear and accurate 
answers strictly using content directly from the PDF. If the answer is not in the provided 
content, simply respond with "I don't know." 

**Important Instructions**:
1. Only use the exact content provided from the document; do not use outside knowledge.
2. Do not summarize the content—provide it exactly as it is.
3. If the content is incomplete or unclear, respond with "The information is incomplete in the document."
4. If the answer is not found in the content, respond with "I don't know."

### Context:
{context}

### User's Question:
{question}

### Response:
"""

# Apply the template
prompt_template = PromptTemplate(input_variables=["context", "question"], template=template)

# Defining Conversation Memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="query", output_key="result")

# Define the RetrievalQA chain with the custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# Query function without memory
def query_pdf_bot(user_input: str) -> str:
    print(f"User Input:\n{user_input}")
    try:
        memory.save_context({"query":user_input}, {"result": ""})
        response = qa_chain.invoke(input=user_input)
        memory.save_context({"query":user_input}, {"result": response['result']})

        chat_history = memory.load_memory_variables({})

       # print("Memory Content: ", memory.load_memory_variables({}))


        # Extract the response and its sources
        answer = response['result']
        source_info = "\n".join([f"- Source: {doc.metadata.get('source', 'Unknown')}" for doc in response['source_documents']])
        
        # Append source information to the answer
        if source_info.strip():
            answer += f"\n\n---\nSources:\n{source_info}"
        
        return {"result": answer, "source_documents": response['source_documents']}, chat_history

    
    except Exception as e:
        print(f"Error in querying LLM: {e}")
        return "Sorry, I couldn't process your request at the moment."
    
if __name__ == "__main__":
    print("GenAI PDF Chatbot is running. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Bot: Exiting. See you next time!")
            break
        bot_response = query_pdf_bot(user_input)
        print(f"Bot: {bot_response}")
