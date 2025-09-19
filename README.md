# MedRAG - Hospital RAG System Demo

## 🏥 Project Overview

MedRAG is an intelligent hospital management system that uses Retrieval-Augmented Generation (RAG) to process medical documents and provide AI-powered assistance to patients and hospital staff. This demo showcases a modern web interface for the complete system.

## ✨ Features

### 👥 Role-Based Access

- **Patient Interface**: Chat-based AI assistant for finding doctor information, appointments, and hospital services
- **Hospital Manager Dashboard**: Analytics, document management, and system oversight
- **Voice Interface**: Speech-to-text interaction for accessibility

### 🤖 AI-Powered Capabilities

- **Document Processing**: Upload and process PDFs and text files containing doctor profiles and medical information
- **Natural Language Queries**: Ask questions in plain English about doctors, appointments, and hospital policies
- **Smart Responses**: Context-aware answers based on uploaded documents
- **Voice Interaction**: Hands-free conversation with the AI assistant

### 📊 Management Features

- **Analytics Dashboard**: Usage statistics, query tracking, and performance metrics
- **Document Management**: Upload, process, and monitor medical documents
- **Query Logging**: Track all patient interactions for quality assurance
- **Real-time Monitoring**: Live statistics and system health metrics

## 🏗️ Technical Architecture

### Frontend Stack

- **React 18** with TypeScript
- **Tailwind CSS** for styling with custom medical theme
- **shadcn/ui** component library
- **React Router** for navigation
- **React Query** for state management

### Planned Backend Integration (Requires Supabase)

- **Document Storage**: Secure file upload and management
- **Vector Database**: Embeddings for semantic search
- **Edge Functions**: AI processing with OpenAI/Azure integration
- **Authentication**: Role-based access control
- **Real-time Features**: Live chat and notifications

## 🚀 Getting Started

### Demo Mode (Current)

The current implementation is a fully functional demo with:

- Beautiful, responsive UI
- Mock data and simulated interactions
- Complete user flows for both patients and managers
- Voice interface simulation

### Production Setup (Requires Supabase)

To implement the full RAG system:

1. **Connect to Supabase**
   - Click the Supabase button in Lovable
   - Set up database tables for documents, users, and queries
   - Configure authentication and RLS policies

2. **Add AI Integration**
   - Configure OpenAI or Azure OpenAI API keys
   - Set up edge functions for document processing
   - Implement vector embeddings for semantic search

3. **Document Processing Pipeline**
   - PDF text extraction
   - Text chunking and embedding generation
   - Vector storage and retrieval

## 📋 Implementation Plan

### Phase 1: Backend Setup ✅ (Demo Complete)

- [x] UI/UX Design System
- [x] Component Architecture
- [x] Routing and Navigation
- [x] Mock Data Integration

### Phase 2: Supabase Integration (Next Steps)

- [ ] Database schema design
- [ ] Authentication system
- [ ] File storage configuration
- [ ] Edge function setup

### Phase 3: AI Integration

- [ ] OpenAI/Azure API integration
- [ ] Document processing pipeline
- [ ] Vector database setup
- [ ] RAG query implementation

### Phase 4: Advanced Features

- [ ] Real-time chat
- [ ] Voice-to-text integration
- [ ] Advanced analytics
- [ ] Mobile responsiveness

## 🔧 Configuration

### Environment Variables (Supabase)

```env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
```

### Document Processing Settings

- **Supported Formats**: PDF, TXT, DOCX
- **Max File Size**: 10MB per document
- **Processing Time**: ~30 seconds per document
- **Embedding Model**: text-embedding-ada-002 (OpenAI) or text-embedding-ada-002 (Azure)

## 🎯 User Flows

### Patient Journey

1. **Access System** → Select "Patient Access"
2. **Chat Interface** → Ask questions about doctors/appointments
3. **Voice Option** → Switch to voice interaction if needed
4. **Get Answers** → Receive AI-powered responses based on hospital documents

### Manager Journey

1. **Admin Access** → Select "Hospital Manager"
2. **Dashboard Overview** → View system statistics and metrics
3. **Document Management** → Upload and process new medical documents
4. **Query Analysis** → Review patient interactions and system performance

## 🔒 Security & Compliance

- **Role-Based Access Control**: Separate interfaces for patients vs managers
- **Data Privacy**: Patient queries are logged anonymously
- **Document Security**: Encrypted storage and processing
- **HIPAA Considerations**: Designed with healthcare privacy in mind

## 📱 Responsive Design

- **Mobile-First**: Optimized for all device sizes
- **Accessibility**: WCAG 2.1 compliant design
- **Performance**: Optimized loading and interactions
- **PWA Ready**: Can be installed as a progressive web app

## 🛠️ Development

### Local Development

```bash
# 1) Install Node deps (root) and Python deps (backend)
npm install
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt || uv pip install -r backend/requirements.txt

# 2) Set environment variables
# Create a .env file at project root (used by Vite) with:
# VITE_SUPABASE_URL=your_supabase_url
# VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
# Optionally set backend env in backend/.env:
# OPENAI_API_KEY=your_openai_api_key
# API_HOST=0.0.0.0
# API_PORT=8000

# 3) Run frontend and backend together
npm run dev
```

### Build for Production

```bash
npm run build
```

### Deploy

Use Lovable's built-in deployment or connect to your preferred hosting platform.

## 🔌 Supabase Integration

- Frontend reads `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` from environment variables (no keys are hardcoded).
- Database schema and RLS policies are provided under `supabase/migrations/` (profiles, documents, doctors, schedules, appointments, chat_logs, document_embeddings, and a private `documents` storage bucket).
- Manager dashboard uses Supabase Storage (bucket `documents`) and the `documents` table for metadata. Ensure storage policies from migrations are applied.
- Authentication and profiles are handled by `src/hooks/useAuth.tsx` using Supabase Auth and a `profiles` table (with roles: patient, staff, manager).

## ▶️ One-Command Dev

The root script `npm run dev` runs both the Vite frontend (port 8080 with API proxy) and the FastAPI backend (port 8000). The proxy forwards `/api`, `/health`, and `/docs` to the backend.

## 📈 Future Enhancements

- **Multi-language Support**: Translate interface and AI responses
- **Advanced Analytics**: Machine learning insights on query patterns
- **Integration APIs**: Connect with existing hospital systems
- **Mobile App**: Native iOS/Android applications
- **Workflow Automation**: Automated appointment scheduling

## 🤝 Contributing

This is a proof-of-concept demo. For production implementation, connect to Supabase and follow the implementation plan above.

## 📄 License

This project is part of a demonstration for hospital management systems. All medical data used is fictional and for demo purposes only.

---

**Ready to implement the full RAG system?** Connect to Supabase to unlock the complete functionality with real document processing, AI integration, and database storage.
