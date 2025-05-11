import streamlit as st
from query import query_pdf_bot

st.set_page_config(page_title = "GenAI PDF Chatbot with Memory")
st.title("Chat with your PDF documents (Memory-Enabled)")

st.write('### Chat History')
if "history" not in st.session_state:
    st.session_state["history"] = []

for entry in st.session_state["history"]:
    st.write(f"**You:** {entry['input']}")
    st.write(f"**Bot:** {entry['output']}")

query = st.text_input("Ask me anything from the PDF:")

if query:
    with st.spinner("Generating response..."):
        response, history = query_pdf_bot(query)

        st.session_state["history"].append({
            "input":query,
            "output":response['result']
        })

        st.write("### Answer:")
        st.write(response['result'])

        if response['source_documents']:
            st.write("### Source Documents:")
            for i, doc in enumerate(response['source_documents']):
                st.write(f"**Document {i+1}:**")
                st.write(doc.metadata)