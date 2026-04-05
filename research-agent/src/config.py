"""Configuration for the Research Agent."""

import os
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

PROJECT_ROOT = Path(__file__).parent.parent
RESEARCH_MEMORY_DIR = PROJECT_ROOT / "workspace" / "research_memory"
RESEARCH_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

GEMINI_MODEL = "gemini-flash-latest"  
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  


CHROMA_COLLECTION_NAME = "research_findings"
CHROMA_DISTANCE_METRIC = "cosine" 


MAX_TOKENS_FOR_RESPONSE = 2000
TEMPERATURE = 0.7  
WEB_SEARCH_MAX_RESULTS = 5
WEB_SEARCH_TIMEOUT = 10


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Set it in .env file or environment variable."
    )

if not TAVILY_API_KEY:
    raise ValueError(
        "TAVILY_API_KEY not found. Set it in .env file or environment variable."
    )
