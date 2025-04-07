# Import necessary libraries
import streamlit as st 
from PyPDF2 import PdfReader  # For reading PDF documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting long texts
import os 
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # To get text embeddings from Google Generative AI
import google.generativeai as genai 
from langchain_community.vectorstores import FAISS  # For storing and retrieving document embeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # Chat model from Google
from langchain.chains.question_answering import load_qa_chain  # For setting up Q&A pipeline
from langchain_core.prompts import PromptTemplate  # For customizing prompt templates

# Function to extract text from uploaded PDF documents
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# Function to split extracted text into overlapping chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks 

# Function to create embeddings for text chunks and store them locally using FAISS
def get_vector_store(text_chunks, api_key):
    genai.configure(api_key=api_key)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index") 

# Function to set up the conversational question-answering chain
def get_conversational_chain(api_key):
    genai.configure(api_key=api_key)

    # Prompt template for controlling the answer format
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". Don't provide the wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Load Google Gemini chat model with low creativity for accurate responses
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.3)
    
    # Use custom prompt and the chat model to create a Q&A chain
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

# Function to handle user questions and display AI-generated answers
def user_input(user_question, api_key):
    genai.configure(api_key=api_key)

    # Load previously stored FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Retrieve the most similar chunks to the question
    docs = new_db.similarity_search(user_question)

    # Get Q&A chain and generate answer
    chain = get_conversational_chain(api_key)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply: ", response["output_text"])

# Utility function to guess legal category based on sample text content
def guess_legal_category(text):
    text = text.lower()
    if "agreement" in text or "party of the first part" in text:
        return "Contract Law"
    elif "intellectual property" in text or "patent" in text:
        return "Intellectual Property Law"
    elif "court" or "judge" or "plaintiff" in text:
        return "Court Judgement / Litigation"
    elif "lease" in text or "tenancy" in text:
        return "Real Estate / Property Law"
    elif "employment" or "termination" in text:
        return "Employment Law"
    else:
        return "General Legal Document"

# Main Streamlit app
def main():
    st.set_page_config(page_title="Legal PDF Analyzer", page_icon='⚖️')

    # Sidebar setup for API key and PDF uploads
    st.sidebar.title("Setup")
    api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
    pdf_docs = st.sidebar.file_uploader("Upload Legal PDF Files", type="pdf", accept_multiple_files=True)

    # Processing button to extract and index document text
    if st.sidebar.button("Submit & Process"):
        if not api_key:
            st.sidebar.error("Please enter your Google API key.")
        elif not pdf_docs:
            st.sidebar.error("Please upload at least one document.")
        else:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.session_state["processed"] = True
                st.session_state["pdf_docs"] = pdf_docs
                st.success("Documents processed successfully!")

    # Two-tab layout
    tab1, tab2 = st.tabs(["📑 Document Overview", "💬 PDF Analyzer (Q&A)"])

    # Tab 1: Summary of uploaded documents
    with tab1:
        st.subheader("Uploaded Document Summary")

        if "pdf_docs" not in st.session_state or not st.session_state["pdf_docs"]:
            st.info("Please upload and process documents from the sidebar.")
        else:
            st.write(f"**Number of Documents:** {len(st.session_state['pdf_docs'])}")
            for doc in st.session_state['pdf_docs']:
                st.markdown(f"- **Filename:** `{doc.name}`")
                try:
                    reader = PdfReader(doc)
                    sample_text = reader.pages[0].extract_text()[:1000]
                    category = guess_legal_category(sample_text)
                    st.write(f"  → Detected Category: *{category}*")
                except Exception as e:
                    st.write("  → Could not analyze this document.")

    # Tab 2: Interactive Q&A system
    with tab2:
        st.subheader("Ask Questions About Your Legal Documents")

        if "processed" not in st.session_state:
            st.info("Please process the documents first.")
        else:
            user_question = st.chat_input("Ask a question related to the uploaded documents...")
            if user_question:
                if not api_key:
                    st.error("Please enter your Google API key.")
                else:
                    user_input(user_question, api_key)

# Run the app
if __name__ == '__main__':
    main()
