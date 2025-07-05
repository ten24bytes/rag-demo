# RAG Demo Application 🔍

A comprehensive **Retrieval-Augmented Generation (RAG)** application built with modern Python technologies. This demo showcases an intelligent document processing and question-answering system using vector similarity search and large language models.

## 🌟 Features

- **📄 Multi-format Document Processing**: Support for PDF, DOCX, PPTX, and TXT files
- **🧠 Intelligent Question Answering**: Powered by DSPy framework with Chain-of-Thought reasoning
- **⚡ High-Performance Vector Search**: Qdrant vector database for fast similarity matching
- **🎯 Configurable Embeddings**: OpenAI text-embedding-3-small with adjustable dimensions
- **🌐 Interactive Web Interface**: Streamlit-based UI with real-time document upload and Q&A
- **🐳 Containerized Deployment**: Docker Compose for easy setup and deployment
- **📊 System Monitoring**: Built-in health checks and status monitoring
- **🔧 Flexible Configuration**: Environment-based settings for all components

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │  Document       │    │   RAG Pipeline  │
│                 │◄──►│  Processor      │◄──►│                 │
│ - File Upload   │    │                 │    │ - DSPy Framework│
│ - Q&A Interface │    │ - PDF/DOCX/PPTX │    │ - ChainOfThought│
│ - Status Monitor│    │ - Text Chunking │    │ - Context Retr. │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │   Vector Store  │              │
         └──────────────┤                 ├──────────────┘
                        │ - Qdrant DB     │
                        │ - Embeddings    │
                        │ - Similarity    │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose**
- **OpenAI API Key** (required for embeddings and LLM)

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-demo
```

### 2. Environment Setup

Copy the example environment file and configure your API keys:

```bash
# Windows
copy .env.example .env

# Linux/macOS
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
```

### 3. Launch the Application

**Windows:**

```cmd
start.bat
```

**Linux/macOS:**

```bash
chmod +x start.sh
./start.sh
```

**Manual Docker Compose:**

```bash
docker-compose up --build
```

### 4. Access the Application

Open your browser and navigate to: **http://localhost:8501**

## 📚 Usage Guide

### Document Upload and Processing

1. **Upload Documents**: Use the sidebar file uploader to select PDF, DOCX, PPTX, or TXT files
2. **Process Documents**: Click "🚀 Process Documents" to chunk and embed your files
3. **Monitor Progress**: Watch the real-time processing status and vector store statistics

### Question Answering

1. **Ask Questions**: Enter questions about your uploaded documents in the main interface
2. **Get Intelligent Answers**: The system retrieves relevant context and generates comprehensive answers
3. **View Chat History**: All Q&A interactions are saved in your session

### System Monitoring

- **Vector Store Status**: Monitor collection health and document count
- **Processing Statistics**: View upload success rates and processing times
- **Real-time Logs**: Check application logs for debugging and monitoring

## 🔧 Configuration

### Environment Variables

| Variable              | Default                  | Description                                |
| --------------------- | ------------------------ | ------------------------------------------ |
| `OPENAI_API_KEY`      | _Required_               | OpenAI API key for embeddings and LLM      |
| `LLM_MODEL`           | `gpt-3.5-turbo`          | OpenAI model for text generation           |
| `EMBEDDING_MODEL`     | `text-embedding-3-small` | OpenAI embedding model                     |
| `EMBEDDING_DIMENSION` | `1536`                   | Embedding vector dimensions (1536/512/256) |
| `QDRANT_URL`          | `http://localhost:6333`  | Qdrant vector database URL                 |
| `CHUNK_SIZE`          | `1000`                   | Text chunk size for processing             |
| `CHUNK_OVERLAP`       | `200`                    | Overlap between text chunks                |

### Embedding Models

The application uses **OpenAI's text-embedding-3-small** for optimal performance:

- **🎯 High Quality**: 61.0% MTEB score
- **💰 Cost Effective**: $0.00002 per 1K tokens
- **⚙️ Configurable Dimensions**:
  - `1536`: Full quality (default)
  - `512`: ~67% cost reduction, minimal quality loss
  - `256`: ~83% cost reduction, moderate quality loss

## 🛠️ Technology Stack

### Core Technologies

- **[DSPy](https://github.com/stanfordnlp/dspy)**: Advanced LLM programming framework
- **[Qdrant](https://qdrant.tech/)**: High-performance vector database
- **[Streamlit](https://streamlit.io/)**: Interactive web application framework
- **[OpenAI](https://openai.com/)**: Embeddings and language models

### Document Processing

- **PyPDF2**: PDF text extraction
- **python-docx**: Microsoft Word document processing
- **python-pptx**: PowerPoint presentation processing

### Infrastructure

- **Docker**: Containerization and deployment
- **uv**: Fast Python package management
- **Loguru**: Structured logging
- **Pydantic**: Configuration and data validation

## 📁 Project Structure

```
rag-demo/
├── 📄 docker-compose.yml      # Multi-service orchestration
├── 🐳 Dockerfile             # Application container
├── ⚙️  pyproject.toml         # Project dependencies
├── 🚀 start.bat / start.sh    # Quick start scripts
├── 📝 README.md               # This file
├── 🔧 .env.example            # Environment template
│
├── 📂 src/                    # Application source code
│   ├── 🎯 app.py              # Streamlit web interface
│   ├── ⚙️  config.py          # Configuration management
│   ├── 📄 document_processor.py # Multi-format document handling
│   ├── 🔗 rag_pipeline.py     # DSPy-powered RAG implementation
│   ├── 🗃️  vector_store.py    # Qdrant vector operations
│   └── 📋 logging_config.py   # Structured logging setup
│
├── 🧪 tests/                  # Test suite
│   └── test_rag_app.py        # Application tests
│
├── 📂 uploads/                # Document upload directory
└── 📂 logs/                   # Application logs
```

## 🧪 Development

### Local Development Setup

1. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   # or using uv
   uv sync
   ```

2. **Start Qdrant Database:**

   ```bash
   docker run -p 6333:6333 qdrant/qdrant:latest
   ```

3. **Run the Application:**
   ```bash
   streamlit run src/app.py
   ```

### Testing

Run the test suite:

```bash
pytest tests/
```

### Code Quality

The project uses modern Python tooling:

- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pytest**: Testing framework

## 🔍 How It Works

### 1. Document Processing Pipeline

```python
# Document → Chunks → Embeddings → Vector Storage
Document → [Text Extraction] → [Chunking] → [Embedding] → [Qdrant Storage]
```

### 2. Question Answering Flow

```python
# Query → Retrieval → Generation → Response
User Query → [Embedding] → [Vector Search] → [Context Retrieval] → [DSPy Chain-of-Thought] → Answer
```

### 3. DSPy Integration

The application leverages **DSPy's Chain-of-Thought** reasoning:

```python
class RAGSignature(dspy.Signature):
    context = dspy.InputField(desc="Retrieved relevant context")
    question = dspy.InputField(desc="User's question")
    answer = dspy.OutputField(desc="Comprehensive answer")

# Chain-of-Thought reasoning for better answers
self.generate_answer = dspy.ChainOfThought(RAGSignature)
```

## 🚀 Deployment

### Production Deployment

1. **Environment Configuration**: Set production environment variables
2. **Container Orchestration**: Use the provided Docker Compose configuration
3. **Resource Scaling**: Adjust container resources based on usage
4. **Monitoring**: Monitor logs and system health endpoints

### Cloud Deployment

The application is cloud-ready and can be deployed on:

- **AWS**: ECS, EKS, or EC2
- **Google Cloud**: Cloud Run, GKE, or Compute Engine
- **Azure**: Container Instances, AKS, or Virtual Machines

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[DSPy Team](https://github.com/stanfordnlp/dspy)** - Revolutionary LLM programming framework
- **[Qdrant](https://qdrant.tech/)** - High-performance vector database
- **[Streamlit](https://streamlit.io/)** - Elegant web app framework
- **[OpenAI](https://openai.com/)** - Advanced language models and embeddings

## 📞 Support

- **Issues**: [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)
- **Documentation**: Check the inline code documentation

---

**Built with ❤️ using modern Python technologies**
