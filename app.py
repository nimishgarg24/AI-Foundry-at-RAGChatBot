import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import ChatPromptTemplate

from langchain.schema.runnable import RunnableMap, RunnablePassthrough

# ----- ENVIRONMENT & LANGCHAIN SETTINGS -----
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# ----- STREAMLIT PAGE LAYOUT -----
st.set_page_config(page_title="StudyMate AI", page_icon=":books:", layout="wide")
st.sidebar.header("StudyMate AI")
st.sidebar.markdown("Your intelligent learning companion")

uploaded_files = st.sidebar.file_uploader(
    "Upload Documents", type=["pdf"], accept_multiple_files=True
)
temperature = st.sidebar.slider(
    "Model Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05
)

st.markdown("""
    <h2 style='text-align: center; color: #6658F6;'>Welcome to <span style='color: #6658F6;'>StudyMate AI</span></h2>
    <p style='text-align: center;'>Your intelligent learning companion powered by advanced AI.<br>
    Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.</p>
    """, unsafe_allow_html=True
)
st.image("https://images.unsplash.com/photo-1513258496099-48168024aec0", use_container_width=True)

# ----- LOAD & EMBED DOCUMENTS -----
def load_and_embed(files):
    texts = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.read())
            tmp_file.flush()
            tmp_path = tmp_file.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        texts.extend([chunk.page_content for chunk in chunks if chunk.page_content.strip()])
    st.info(f"Chunked to {len(texts)} text blocks.")
    embedder = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts, embedder)
    return vector_store

# ----- RAG PROMPT TEMPLATE -----
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {question}
    """
)

# ----- MODERN RAG CHAIN USING PIPE SYNTAX -----
def get_rag_response(query, vector_store, temperature, prompt):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model="gemma2:2b", temperature=temperature)
    # Compose the modular RAG chain
    rag_chain = (
        RunnableMap({
            "context": retriever,
            "question": RunnablePassthrough()
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke(query)
    return answer

# ----- MAIN LOGIC -----
vector_store = None
if uploaded_files:
    vector_store = load_and_embed(uploaded_files)
    st.success("Documents uploaded and knowledge base created!")

question = st.text_input("Ask me anything about your studies...", value="")
if st.button("Send"):
    if not vector_store:
        st.error("Please upload documents to begin.")
    elif not question.strip():
        st.info("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            answer = get_rag_response(
                question,
                vector_store,
                temperature,
                prompt
            )
            # Show only the final answer, nothing else
            st.markdown(
                f"<div style='margin-bottom: 20px;'><b>StudyMate AI:</b> {answer}</div>",
                unsafe_allow_html=True
            )

