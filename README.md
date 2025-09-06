# PDF Q&A Chatbot

A Python application that allows you to ask questions about PDF documents using Google's Gemini AI and vector similarity search.

## Features

- Extract text from PDF files
- Split documents into manageable chunks
- Create vector embeddings using HuggingFace transformers
- Perform similarity search to find relevant context
- Generate answers using Google's Gemini 1.5 Flash model
- Interactive chat interface

## Requirements

- Python 3.7+
- Google API Key for Gemini

## Installation

1. Clone or download this repository
2. Install required dependencies:
```bash
pip install python-dotenv google-generativeai PyPDF2 langchain langchain-community faiss-cpu sentence-transformers
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Enter the path to your PDF file when prompted
3. Wait for the system to process the document (extract text, create chunks, build vector store)
4. Start asking questions about your document
5. Type "exit" to quit the application

## How It Works

1. **Text Extraction**: Uses PyPDF2 to extract text from PDF files
2. **Text Chunking**: Splits large documents into smaller chunks using RecursiveCharacterTextSplitter
3. **Vector Store**: Creates embeddings using HuggingFace's sentence-transformers model and stores them in FAISS
4. **Question Answering**: Performs similarity search to find relevant context and uses Google's Gemini model to generate answers

## Dependencies

- `python-dotenv`: Environment variable management
- `google-generativeai`: Google Gemini AI integration
- `PyPDF2`: PDF text extraction
- `langchain`: Text processing and prompt templates
- `langchain-community`: Vector stores and embeddings
- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Text embeddings

## Error Handling

The application includes error handling for:
- Missing PDF files
- PDF reading errors
- Missing API keys
- AI model generation errors

## Notes

- The application uses the "sentence-transformers/all-MiniLM-L6-v2" model for embeddings
- Gemini 1.5 Flash is used for fast and capable response generation
- Text chunks are limited to 1000 characters with 200 character overlap
- Similarity search returns the top 3 most relevant chunks
