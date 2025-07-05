@echo off
REM RAG Demo Startup Script for Windows

echo 🚀 Starting RAG Demo Application...

REM Check if Docker is running
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not installed or not running
    pause
    exit /b 1
)

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker Compose is not installed
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist .env (
    echo ⚠️  .env file not found, copying from .env.example
    copy .env.example .env
    echo 📝 Please edit .env file with your API keys before continuing
    echo    Required: OPENAI_API_KEY
    pause
    exit /b 1
)

echo ✅ Configuration validated

REM Create uploads directory
if not exist uploads mkdir uploads

REM Start the application
echo 🐳 Starting containers...
docker-compose up --build

echo 🎉 RAG Demo Application is ready!
echo 🌐 Open your browser and go to: http://localhost:8501
pause
