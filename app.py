import os
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

def get_pdf_text(pdf_path):
    """
    Extracts text from a given PDF file.

    Args:
        pdf_path (str): The file path to the PDF.

    Returns:
        str: The concatenated text from all pages of the PDF.
    """
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        return None
    return text

def get_text_chunks(text):
    """
    Splits a long text into smaller chunks.

    Args:
        text (str): The input text.

    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks.

    Args:
        text_chunks (list): A list of text chunks.

    Returns:
        FAISS: A FAISS vector store object.
    """
    # Using a popular open-source embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain():
    """
    Creates a conversational prompt and configures the Gemini model.

    Returns:
        genai.GenerativeModel: The configured Generative Model.
        str: The prompt template string.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context.
    If the answer is not in the provided context, just say, "The answer is not available in the context".
    Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    # Using the latest Flash model which is fast and capable for RAG
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    return model, prompt_template

def user_input(user_question, vector_store):
    """
    Handles user input, retrieves relevant context, and generates an answer.

    Args:
        user_question (str): The question asked by the user.
        vector_store (FAISS): The vector store containing the document embeddings.
    """
    # Get the embedding model used by the vector store
    # Note: FAISS does not directly store the embedding function, so we re-instantiate it.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Perform similarity search to find relevant documents
    docs = vector_store.similarity_search(user_question, k=3)

    # Get the conversational model and prompt template
    model, prompt_template_str = get_conversational_chain()

    # Format the prompt with the retrieved context and user question
    prompt = PromptTemplate(template=prompt_template_str, input_variables=["context", "question"])
    formatted_prompt = prompt.format(context="\n".join([doc.page_content for doc in docs]), question=user_question)

    # Generate the response from the LLM
    try:
        response = model.generate_content(formatted_prompt)
        print("\n🤖 Reply:")
        print(response.text)
    except Exception as e:
        print(f"An error occurred while generating the response: {e}")


def main():
    """
    The main function to run the RAG application.
    """
    # Configure the Google AI SDK
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found. Please set it in the .env file.")
        return
    genai.configure(api_key=api_key)

    print("🚀 Welcome to the PDF Q&A Chatbot!")

    pdf_path = input("Enter the path to your PDF file: ").strip()

    # Step 1: Extract text from PDF
    print("\n1️⃣  Extracting text from the PDF...")
    raw_text = get_pdf_text(pdf_path)
    if not raw_text:
        return # Exit if PDF processing failed

    # Step 2: Split text into chunks
    print("2️⃣  Splitting text into manageable chunks...")
    text_chunks = get_text_chunks(raw_text)

    # Step 3: Create the vector store
    print("3️⃣  Creating vector store (this might take a moment)...")
    vector_store = get_vector_store(text_chunks)

    print("\n✅ Setup complete! You can now ask questions about your document.")

    while True:
        user_question = input("\n🙋 You: ").strip()
        if user_question.lower() == "exit":
            print("\n👋 ok Goodbye, I am here whenever you want!")
            break
        elif user_question:
            user_input(user_question, vector_store)

if __name__ == "__main__":
    main()

