"""
Ollama Local REST & API Client.
Provides methods to inspect models, pull weights, check health, and interact with Ollama.
"""

import json
import logging
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Client for interacting with Ollama's native HTTP REST API.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ):
        if base_url:
            self.base_url = base_url.rstrip("/")
            # Normalize if base_url includes /v1
            if self.base_url.endswith("/v1"):
                self.base_url = self.base_url[:-3]
        else:
            self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        url = f"{self.base_url}{endpoint}"
        req_timeout = timeout or self.timeout
        req_data = None
        headers = {"Content-Type": "application/json"}

        if data is not None:
            req_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=req_timeout) as response:
                content = response.read().decode("utf-8")
                if not content.strip():
                    return {}
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return content
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="ignore")
            logger.error("Ollama HTTP error %s for %s: %s", e.code, url, error_body)
            raise RuntimeError(f"Ollama API error ({e.code}) on {endpoint}: {error_body}") from e
        except Exception as e:
            logger.debug("Failed request to %s: %s", url, e)
            raise

    def ping(self) -> bool:
        """Returns True if the Ollama endpoint is reachable."""
        try:
            self.get_version()
            return True
        except Exception:
            return False

    def get_version(self) -> str:
        """Retrieves Ollama server version."""
        res = self._request("/api/version", method="GET")
        if isinstance(res, dict):
            return res.get("version", "unknown")
        return str(res)

    def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists all locally downloaded models via /api/tags.
        """
        res = self._request("/api/tags", method="GET")
        if isinstance(res, dict) and "models" in res:
            return res["models"]
        return []

    def list_model_names(self) -> List[str]:
        """
        Returns a list of model names installed in Ollama.
        """
        models = self.list_models()
        names = []
        for m in models:
            name = m.get("name") or m.get("model")
            if name:
                names.append(name)
        return names

    def is_model_available(self, model_name: str) -> bool:
        """
        Checks whether a given model name is installed.
        Matches exact tag or tag without ':latest'.
        """
        installed = self.list_model_names()
        target = model_name.strip().lower()
        for name in installed:
            name_lower = name.lower()
            if name_lower == target:
                return True
            # Check prefix match if no tag specified
            if ":" not in target and name_lower == f"{target}:latest":
                return True
            if name_lower.startswith(f"{target}:"):
                return True
        return False

    def pull_model(self, model_name: str, stream: bool = False, timeout: float = 600.0) -> bool:
        """
        Pulls a model from the Ollama library.
        """
        logger.info("Pulling model '%s' via Ollama API (timeout=%ss)...", model_name, timeout)
        try:
            res = self._request(
                "/api/pull",
                method="POST",
                data={"name": model_name, "stream": stream},
                timeout=timeout,
            )
            if isinstance(res, dict) and res.get("status") == "success":
                logger.info("Model '%s' successfully pulled.", model_name)
                return True
            logger.info("Pull response for '%s': %s", model_name, res)
            return True
        except Exception as e:
            logger.error("Failed to pull model '%s': %s", model_name, e)
            return False

    def show_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieves details and modelfile information for a specific model.
        """
        return self._request("/api/show", method="POST", data={"name": model_name})

    def delete_model(self, model_name: str) -> bool:
        """
        Deletes a model from the local Ollama instance.
        """
        try:
            self._request("/api/delete", method="DELETE", data={"name": model_name})
            return True
        except Exception as e:
            logger.error("Failed to delete model '%s': %s", model_name, e)
            return False

