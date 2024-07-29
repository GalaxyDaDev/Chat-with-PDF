import streamlit as st
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
import json
from streamlit_lottie import st_lottie
import time

os.environ["GOOGLE_API_KEY"] = "AIzaSyDS7yo81FTHnivuZzno6EeXwgtLG9vrs44"

def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def get_vectorstore_from_PDF(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context: If you cannot find relevant context use google to get most relevant context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),    
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

def wrap_text(text, max_width=80):
    return '<br>'.join(textwrap.wrap(text, max_width))

def render_chat_history():
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                st.markdown(f"""
                    <div class="chat-message user fade-in">
                        <div class="message-container">
                            <div class="message-content">{message.content}</div>
                        </div>
                        <div class="avatar"><i class="fas fa-user"></i></div>
                    </div>
                """, unsafe_allow_html=True)
            elif isinstance(message, AIMessage):
                st.markdown(f"""
                    <div class="chat-message bot fade-in">
                        <div class="avatar"><i class="fas fa-robot"></i></div>
                        <div class="message-container">
                            <div class="message-content">{message.content}</div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

st.set_page_config(page_title="ChatPDF", page_icon="💬", layout="wide")

# Updated CSS for dynamic message boxes
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

body {
    font-family: 'Arial', sans-serif;
    background-color: #f0f2f6;
    color: #333;
}

.fade-in {
    animation: fadeIn 0.5s;
}
@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}
.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}
.chat-message {
    display: flex;   
    align-items: flex-start;
    margin-bottom: 1rem;
    padding: 0.5rem;
}
.chat-message.user {
    justify-content: flex-end;
}
.chat-message.bot {
    justify-content: flex-start;
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    color: #fff;
    flex-shrink: 0;
}
.chat-message.user .avatar {
    background-color: #128C7E;
    margin-left: 10px;
    order: 1;
}
.chat-message.bot .avatar {
    background-color: #075E54;
    margin-right: 10px;
}
.message-content {
    padding: 0.5rem 1rem;
    border-radius: 0.5rem;
    word-wrap: break-word;
}
.chat-message.user .message-content {
    background: linear-gradient(to bottom left, #09203f, #537895);
    color: #ffffff;
}
.chat-message.bot .message-content {
    background: linear-gradient(to top right, #000428, #004e92);
    color: #ffffff;
}
.message-container {
    display: inline-block;
    max-width: 70%;
}
.chat-message.user .message-container {
    text-align: right;
}
.chat-message.bot .message-container {
    text-align: left;
}
.stTextInput > div > div > input {
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("Chat with PDF 🗣️📑")

with st.sidebar:
    st.header("Upload here ⬇️")
    pdf = st.file_uploader("", type="pdf")

if pdf is None:
    left_column, right_column = st.columns(2)

    with left_column:
        st.markdown("""
            #### 💡 **Why ChatPDF ?**
            1. Targeted Information: Extracts precise, context-relevant details from uploaded PDFs.
            2. Quick Responses: Provides fast, accurate answers tailored to your documents.
            3. Easy to Use: Features an intuitive, user-friendly interface.
            4. Ideal for Researchers: Perfect for students and researchers needing instant, detailed information.
            """)
        st.markdown("""
            #### 📚 **Features**
            1. Get instant answers to your queries.
            2. Enjoy a seamless and interactive document exploration.
            3. Enhance your learning and productivity.
                    """)
        st.markdown("""
            #### 🚀 **Getting Started**
            1. Upload your PDF file.
            2. Type your desired question.
            3. Receive instant, context-aware responses from our AI-powered chatbot.
        """)
          
    with right_column:
        st.markdown("""
            #### 🎯 **Instructions**:
            1. Use the sidebar to upload your PDF file.
            2. Only One PDF can be uploaded at a time. 
            3. You can type your questions or queries in the chatbox below.
            4. The bot will provide answers based on the content of the uploaded PDF. 
        """)
        st.markdown("""
            #### 🤖 **Support:**
            For any assistance, you can contact me
            1. Email: himanshu.cosmicstudios@gmail.com  
            2. Phone No: 9717165887
            """)

else:
    # Create a placeholder for the Lottie animation
    lottie_placeholder = st.empty()

    # Show the Lottie animation while processing
    with lottie_placeholder:
        lottie_animation = load_lottie_file("Animation - 1719938774459.json")
        st_lottie(lottie_animation, speed=1, height=600, key="pdf_upload")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_PDF(pdf)
    
    # Clear the Lottie animation once processing is complete
    lottie_placeholder.empty()

    render_chat_history()

    user_query = st.chat_input("Type your message here...", key="user_input")
    if user_query:
        with st.spinner("Generating..."):
            # Append user query
            st.session_state.chat_history.append(HumanMessage(content=user_query))
            
            # Generate and append bot response
            response = get_response(user_query)
            st.session_state.chat_history.append(AIMessage(content=response))
        
        # Force a rerun to update the UI
        st.rerun()

