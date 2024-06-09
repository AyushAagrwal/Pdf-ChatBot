import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import warnings
import fitz  # PyMuPDF
from textblob import TextBlob



warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Read PDF file and return text
def get_pdf_text(pdf_doc):
    text = ""
    try:
        pdf_reader = PdfReader(pdf_doc)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                raise IndexError("No text could be extracted from this page.")
    except IndexError as e:
        st.session_state.error_message = f"Error extracting text from {pdf_doc.name}: {e}"
        return None
    except Exception as e:
        st.session_state.error_message = f"An unexpected error occurred while reading {pdf_doc.name}: {e}"
        return None
    return text

# Convert PDF to text using PyMuPDF (fitz)
def pdf_to_text(pdf_path, txt_path):
    document = fitz.open(pdf_path)
    with open(txt_path, 'w', encoding='utf-8') as text_file:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text = page.get_text()
            text_file.write(text)
            text_file.write('\n' + '-'*80 + '\n')
    print(f'Text extracted from {pdf_path} and saved to {txt_path}')
    return txt_path


# Fallback method to handle PDF extraction errors
def get_pdf_text_with_fallback(pdf_doc):
    raw_text = get_pdf_text(pdf_doc)
    if raw_text is None:
        # Fallback to PyMuPDF extraction
        st.write("Error extracting text from PDF. Trying an alternative method, please wait...")
        try:
            with open(f"temp_{pdf_doc.name}", "wb") as f:
                f.write(pdf_doc.getbuffer())
            txt_path = f"temp_{pdf_doc.name}.txt"
            pdf_to_text(f"temp_{pdf_doc.name}", txt_path)
            with open(txt_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
            if not raw_text.strip():
                st.session_state.error_message = "No text extracted from the uploaded PDF."
                return None
        except Exception as e:
            st.session_state.error_message = f"An error occurred during fallback extraction: {e}"
            return None
    return raw_text


# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# Get embeddings for each chunk
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to correct spelling mistakes in user input
def correct_spelling(user_input):
    # Create a TextBlob object
    blob = TextBlob(user_input)
    
    # Correct the spelling
    corrected_text = blob.correct()
    
    return str(corrected_text)



# Create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say "answer is not available in the context". Do not provide a wrong answer.

    Context:\n{context}\n
    Question:\n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   client=genai,
                                   temperature=0.3)
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload a PDF and ask me a question"}]

# Handle user input and provide a response
def user_input(user_question):
    # Correct spelling of user input
    corrected_question = correct_spelling(user_question)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(corrected_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": corrected_question}, return_only_outputs=True)
    return response

# Main function to run the Streamlit app
def main():
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="🤖"
    )

    # Load custom CSS
    # load_css()

    # Upload PDF file
    st.title("Upload a PDF file")
    pdf_doc = st.file_uploader("Upload your PDF file", type="pdf")

    if pdf_doc:
        st.write("PDF file uploaded successfully!")
        st.session_state.input_disabled = False

        # Process PDF text
        with st.spinner("Processing PDF..."):
            raw_text = get_pdf_text_with_fallback(pdf_doc)
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processed successfully!")

                # Reset error message
                st.session_state.error_message = None
            else:
                st.session_state.input_disabled = True


    # Display error message if any
    if st.session_state.get("error_message"):
        st.error(st.session_state.error_message)
        # st.write(f"<div class='error-message'>{st.session_state.error_message}</div>", unsafe_allow_html=True)

    # Main content area for displaying chat messages
    st.title("Chat with PDF files")
    st.write("Welcome to the chat!")
    st.button('Clear Chat History', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload a PDF and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if pdf_doc and not st.session_state.input_disabled:
        prompt = st.chat_input("Type your question here")

        if prompt:
            # Correct spelling of user input
            corrected_prompt = correct_spelling(prompt)
            st.session_state.messages.append({"role": "user", "content": corrected_prompt})
            with st.chat_message("user"):
                st.write(corrected_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(corrected_prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
            if response is not None:
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
