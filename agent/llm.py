"""
Thin wrapper around the Ollama local API.

Ollama runs models like llama3.2, mistral, qwen2.5, gemma2 locally
on your machine. No API key needed, completely free.

Install:  https://ollama.com/download
Pull a model:  ollama pull llama3.2

This module handles both regular calls (planner) and streaming (answerer).
"""

import json
import logging
import requests
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def _chat_url(self):
        return f"{self.base_url}/api/chat"

    def chat(self, messages: List[Dict], system: str = "", temperature: float = 0.3) -> str:
        """
        Blocking call - returns the full response as a string.
        Used by the planner which needs the full JSON before parsing.
        """
        payload = {
            "model": self.model,
            "messages": _build_messages(messages, system),
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(self._chat_url(), json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except requests.Timeout:
            raise RuntimeError(
                f"Ollama timed out. Is '{self.model}' pulled? Run: ollama pull {self.model}"
            )
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Make sure it's running: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama chat error: {e}")

    def stream(self, messages: List[Dict], system: str = "", temperature: float = 0.3) -> Generator[str, None, None]:
        """
        Streaming call - yields text chunks as they come in.
        Ollama streams NDJSON, one JSON object per line.
        """
        payload = {
            "model": self.model,
            "messages": _build_messages(messages, system),
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(
                self._chat_url(),
                json=payload,
                stream=True,
                timeout=180,   # longer timeout for streaming
            )
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue   # skip malformed lines

        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Make sure it's running: ollama serve"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama stream error: {e}")

    def list_models(self) -> List[str]:
        """Returns locally available model names."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        """Quick health check - returns False if Ollama isn't running."""
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=3)
            return True
        except Exception:
            return False


def _build_messages(messages: List[Dict], system: str) -> List[Dict]:
    """Prepends system message if provided, then appends user messages."""
    out = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend(messages)
    return out
