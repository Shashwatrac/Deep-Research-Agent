import os
from dotenv import load_dotenv

load_dotenv()

# search
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")

# local LLM via Ollama - change OLLAMA_MODEL to whatever you have pulled
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = os.getenv("OLLAMA_MODEL", "llama3.2")

# agent tuning
MAX_SEARCH_RESULTS: int = 8
MAX_SEARCH_QUERIES: int = 4
MAX_FETCH_URLS:     int = 5
MAX_CONTEXT_CHARS:  int = 12000   # keep this reasonable for local models
MAX_HISTORY_TURNS:  int = 6

# persistence
SESSION_DB: str = "sessions.db"

# fetch
FETCH_TIMEOUT:   int = 15
FETCH_MAX_CHARS: int = 30000
