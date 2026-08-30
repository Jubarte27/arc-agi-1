"""
Integration layer between local Ollama and the ARC-CEGIS framework.
"""

import logging
from typing import Any, Optional, Tuple

from .. import config
from .client import OllamaClient
from .server import OllamaServer, ensure_server_running

logger = logging.getLogger(__name__)


def setup_local_ollama(
    model: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 11434,
    models_dir: Optional[str] = None,
    auto_start: bool = True,
    auto_pull: bool = True,
    set_global_config: bool = True,
) -> Tuple[Optional[OllamaServer], OllamaClient]:
    """
    Sets up the local Ollama environment for ARC-CEGIS experiments.

    Args:
        model: Target model name (e.g. 'qwen2.5-coder:7b').
        host: Ollama server host.
        port: Ollama server port.
        models_dir: Optional custom directory for models.
        auto_start: Whether to automatically launch the server if not running.
        auto_pull: Whether to automatically pull the target model if not present.
        set_global_config: Whether to update arc_cegis.config globally.

    Returns:
        Tuple of (OllamaServer instance or None, OllamaClient instance).
    """
    client = OllamaClient(host=host, port=port)
    server: Optional[OllamaServer] = None

    if auto_start:
        if not client.ping():
            logger.info("Local Ollama server not detected. Starting server at %s:%s...", host, port)
            server = ensure_server_running(host=host, port=port, models_dir=models_dir)
        else:
            logger.info("Connected to existing local Ollama server at %s:%s", host, port)

    target_model = model or config.MODEL_NAME
    if auto_pull and target_model and client.ping():
        if not client.is_model_available(target_model):
            logger.info("Model '%s' not found locally. Pulling now...", target_model)
            client.pull_model(target_model)
        else:
            logger.debug("Model '%s' is already available locally.", target_model)

    if set_global_config:
        openai_endpoint = f"http://{host}:{port}/v1"
        config.LLM_PROVIDER = "ollama"
        config.API_BASE_URL = openai_endpoint
        if model:
            config.MODEL_NAME = model
        # Local models have no cloud RPM/RPD limits by default
        if config.REQUEST_DELAY == 4.2:
            config.REQUEST_DELAY = 0.0
        logger.info(
            "Configured ARC-CEGIS for local Ollama: provider=%s, model=%s, endpoint=%s",
            config.LLM_PROVIDER,
            config.MODEL_NAME,
            config.API_BASE_URL,
        )

    return server, client


def get_local_openai_client(base_url: Optional[str] = None) -> Any:
    """
    Instantiates an OpenAI client configured for the local Ollama endpoint.
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("The OpenAI SDK is required. Run: pip install openai") from e

    url = base_url or config.get_api_base_url("ollama") or "http://127.0.0.1:11434/v1"
    return OpenAI(base_url=url, api_key="ollama")

