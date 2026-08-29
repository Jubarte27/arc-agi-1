"""
Local Ollama Execution & Integration Subpackage for ARC-CEGIS.
"""

from .client import OllamaClient
from .integration import get_local_openai_client, setup_local_ollama
from .server import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    OllamaServer,
    ensure_server_running,
    find_ollama_binary,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "OllamaServer",
    "OllamaClient",
    "ensure_server_running",
    "find_ollama_binary",
    "setup_local_ollama",
    "get_local_openai_client",
]

