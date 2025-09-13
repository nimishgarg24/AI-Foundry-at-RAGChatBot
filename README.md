# STUDY-AI-RAG-BASED-CHATBOT

# StudyMate AI 📚

Your intelligent learning companion powered by advanced AI. Upload your study materials, ask questions, and get personalized explanations tailored to your learning style.

## Features

- **PDF Document Processing**: Upload multiple PDF documents for analysis
- **Intelligent Q&A**: Ask questions about your uploaded documents and get accurate, context-based answers
- **Adjustable Temperature**: Fine-tune the AI's creativity and randomness in responses
- **Modern UI**: Clean, user-friendly Streamlit interface
- **Vector Search**: Efficient document retrieval using FAISS vector database
- **Local AI Model**: Uses Ollama's Gemma2:2b model for privacy and offline capability

## Prerequisites

Before running StudyMate AI, ensure you have the following installed:

- Python 3.8 or higher
- Ollama with the Gemma2:2b model
- Required Python packages (see Installation section)

### Installing Ollama and Gemma2:2b

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull the Gemma2:2b model:
   ```bash
   ollama pull gemma2:2b
   ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd studymate-ai
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install streamlit langchain langchain-community faiss-cpu sentence-transformers pypdf python-dotenv
   ```

4. **Set up environment variables** (optional):
   Create a `.env` file in the project root:
   ```env
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_PROJECT=your_project_name
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://localhost:8501`

3. **Upload documents**:
   - Use the sidebar file uploader to upload one or more PDF documents
   - Supported format: PDF files only

4. **Adjust settings**:
   - Use the temperature slider to control response creativity (0.0 = focused, 1.0 = creative)

5. **Ask questions**:
   - Enter your question in the text input field
   - Click "Send" to get AI-powered answers based on your documents

## How It Works

1. **Document Processing**: PDFs are loaded and split into manageable chunks using RecursiveCharacterTextSplitter
2. **Embedding Creation**: Text chunks are converted to vector embeddings using SentenceTransformer
3. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient retrieval
4. **Query Processing**: User questions are processed through a RAG (Retrieval-Augmented Generation) pipeline
5. **Context Retrieval**: Relevant document chunks are retrieved based on semantic similarity
6. **Answer Generation**: The Ollama Gemma2:2b model generates contextual answers

## Project Structure

```
studymate-ai/
│
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (optional)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Dependencies

- `streamlit`: Web application framework
- `langchain`: Framework for LLM applications
- `langchain-community`: Community extensions for LangChain
- `faiss-cpu`: Vector database for similarity search
- `sentence-transformers`: Embedding models
- `pypdf`: PDF processing library
- `python-dotenv`: Environment variable management

## Configuration Options

### Model Temperature
- **Range**: 0.0 - 1.0
- **Low values (0.0-0.3)**: More focused, deterministic responses
- **High values (0.7-1.0)**: More creative, varied responses
- **Default**: 0.2

### Retrieval Settings
- **Chunk size**: 1000 characters
- **Chunk overlap**: 200 characters
- **Retrieved chunks**: Top 5 most relevant

## Troubleshooting

### Common Issues

1. **Ollama model not found**:
   ```bash
   ollama pull gemma2:2b
   ```

2. **FAISS installation issues**:
   ```bash
   pip install faiss-cpu
   # or for GPU support:
   pip install faiss-gpu
   ```

3. **PDF loading errors**:
   - Ensure PDF files are not corrupted
   - Check file permissions
   - Try with different PDF files

4. **Memory issues with large documents**:
   - Reduce chunk size in the text splitter
   - Process fewer documents at once
   - Consider using a more powerful machine

## Performance Tips

- **Optimal chunk size**: 1000-1500 characters work well for most documents
- **Document size**: Process documents in batches for better performance
- **Hardware**: More RAM helps with larger document collections

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **LangChain**: For the excellent framework for building LLM applications
- **Ollama**: For providing easy-to-use local LLM deployment
- **Streamlit**: For the intuitive web application framework
- **FAISS**: For efficient vector similarity search
- **Sentence Transformers**: For high-quality text embeddings

## Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Check the documentation of the underlying libraries

---

**Happy studying with StudyMate AI! 🎓**
