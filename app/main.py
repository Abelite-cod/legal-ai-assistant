from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.db import engine, Base
from app.models.db_models import Conversation, Message

# Load environment variables
load_dotenv()

# Create DB tables
Base.metadata.create_all(bind=engine)

# Initialize app
app = FastAPI(
    title="Legal AI RAG API",
    description="Conversational Legal Assistant with RAG + Gemini"
)

# Enable CORS (for HTML / frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(routes.router)