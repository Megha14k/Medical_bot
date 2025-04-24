import os
import re
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="wide")

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def main():
    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>AI Chatbot</h1>", unsafe_allow_html=True)
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role_class = "user" if message['role'] == 'user' else "assistant"
        st.markdown(f"<div class='stChatMessage {role_class}'>{message['content']}</div>", unsafe_allow_html=True)
    
    prompt = st.text_input("Type your question here:")

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        st.markdown(f"<div class='stChatMessage user'>{prompt}</div>", unsafe_allow_html=True)
        
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer the user's question.
        If you don't know the answer, just say that you don't know. Don't make up an answer.
        Don't provide anything outside the given context.
        
        Context: {context}
        Question: {question}
        
        Start the answer directly. No small talk please.
        """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]

            # Improved HTML tag cleaning
            clean_result = re.sub(r"<[^>]*>", "", result).strip()
            result_to_show = clean_result

            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})
            st.markdown(f"<div class='stChatMessage assistant'>{result_to_show}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
