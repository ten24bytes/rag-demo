#!/bin/bash

# RAG Demo Startup Script

echo "🚀 Starting RAG Demo Application..."

# Check if Docker is running
if ! docker --version &> /dev/null; then
    echo "❌ Docker is not installed or not running"
    exit 1
fi

# Check if Docker Compose is available
if ! docker-compose --version &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found, copying from .env.example"
    cp .env.example .env
    echo "📝 Please edit .env file with your API keys before continuing"
    echo "   Required: OPENAI_API_KEY"
    exit 1
fi

# Validate required environment variables
source .env
if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Please set your OPENAI_API_KEY in the .env file"
    exit 1
fi

echo "✅ Configuration validated"

# Create uploads directory
mkdir -p uploads

# Start the application
echo "🐳 Starting containers..."
docker-compose up --build

echo "🎉 RAG Demo Application is ready!"
echo "🌐 Open your browser and go to: http://localhost:8501"
