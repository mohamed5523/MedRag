# MedRag - Intelligent Hospital RAG System 🏥

MedRag is a sophisticated hospital management system that leverages Retrieval-Augmented Generation (RAG) to provide AI-powered assistance to patients and hospital staff. The system integrates modern web technologies with advanced AI capabilities to streamline hospital operations and enhance patient experience.

## ✨ Key Features

### 👥 Role-Based Access
- **Patient Interface**: 
    - AI-powered chat assistant for medical inquiries
    - Voice interaction support (Speech-to-Text & Text-to-Speech)
    - WhatsApp integration for seamless communication
- **Hospital Manager Dashboard**:
    - Comprehensive analytics and reporting
    - Document management system
    - Real-time monitoring of system performance

### 🤖 AI Capabilities
- **Intelligent RAG**: Context-aware responses based on uploaded medical documents
- **Multi-Modal Interaction**: Support for text and voice commands
- **Advanced NLP**: High-accuracy natural language understanding for Egyptian Arabic and English
- **TTS Normalization**: Specialized text-to-speech normalization for natural-sounding Arabic output

### 📊 Management Tools
- **Analytics Dashboard**: Visual insights into query patterns and system usage
- **Document Processing**: Automated ingestion and vectorization of PDF and text documents
- **Query Logging**: Detailed logs for quality assurance and system improvement

## 🏗️ Technical Architecture

### Frontend
- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS with custom medical theme
- **UI Components**: shadcn/ui library
- **State Management**: React Query
- **Routing**: React Router

### Backend
- **API Framework**: FastAPI (Python)
- **Database**: 
    - Supabase (PostgreSQL) for relational data
    - ChromaDB (local) for vector embeddings
- **AI/ML**:
    - OpenAI/Azure OpenAI for LLM and embeddings
    - ElevenLabs/OpenAI for Text-to-Speech
    - Whisper for Speech-to-Text

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (3.10+)
- Docker & Docker Compose (optional)

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mohamed5523/MedRag.git
   cd MedRag
   ```

2. **Backend Setup**
   ```bash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Environment Configuration**
   - Create a `.env` file in the root directory (see `.env.example` if available)
   - Configure necessary API keys (OpenAI, Supabase, ElevenLabs)

5. **Run the Application**
   ```bash
   # Terminal 1: Backend
   cd backend
   uvicorn app.main:app --reload --port 8000

   # Terminal 2: Frontend
   cd frontend
   npm run dev
   ```

### 🐳 Docker Deployment

The project includes a `docker-compose.yml` for easy deployment.

```bash
docker-compose up -d --build
```

## 🔧 Configuration

### Environment Variables
Key environment variables required for the system:

```env
OPENAI_API_KEY=your_key
SUPABASE_URL=your_url
SUPABASE_KEY=your_key
ELEVENLABS_API_KEY=your_key
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed for use as a demonstration of hospital management systems. All medical data used is fictional.
