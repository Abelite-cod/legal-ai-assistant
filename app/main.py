from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

from app.api import routes
from app.db import engine, Base
from app.models.db_models import Conversation, Message

# Load environment variables
load_dotenv()

# Create DB tables
Base.metadata.create_all(bind=engine)

# Initialize app
app = FastAPI(
    title="Legal AI Assistant",
    description="Conversational Legal Assistant with RAG + Gemini",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router)

# Serve the frontend HTML at root
@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
