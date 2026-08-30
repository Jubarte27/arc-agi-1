"""
Ollama Local Server Process Lifecycle Manager.
Supports starting, stopping, monitoring, and auto-discovering local Ollama instances.
"""

import atexit
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

# Default host and port configuration
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434


def find_ollama_binary(custom_path: Optional[str] = None) -> Optional[str]:
    """
    Discovers the path to the Ollama binary.
    Searches in:
      1. Provided custom_path
      2. OLLAMA_BIN environment variable
      3. Project-local .ollama/bin/ollama
      4. Project-local bin/ollama
      5. User home .local/bin/ollama
      6. System PATH via shutil.which('ollama')
    """
    candidates = []
    if custom_path:
        candidates.append(custom_path)

    env_bin = os.getenv("OLLAMA_BIN")
    if env_bin:
        candidates.append(env_bin)

    # Search relative to workspace / current file location
    base_dir = Path(__file__).resolve().parent.parent.parent
    candidates.append(str(base_dir / ".ollama" / "bin" / "ollama"))
    candidates.append(str(base_dir / "bin" / "ollama"))
    candidates.append(str(Path.home() / ".local" / "bin" / "ollama"))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    system_bin = shutil.which("ollama")
    if system_bin:
        return system_bin

    return None


class OllamaServer:
    """
    Manages the lifecycle of a local Ollama server process (`ollama serve`).
    """

    def __init__(
        self,
        binary_path: Optional[str] = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        models_dir: Optional[str] = None,
        keep_alive: Optional[str] = None,
        log_file: Optional[str] = None,
        auto_shutdown: bool = True,
    ):
        self.host = host
        self.port = port
        self.models_dir = models_dir or os.getenv("OLLAMA_MODELS")
        self.keep_alive = keep_alive or os.getenv("OLLAMA_KEEP_ALIVE")
        self.log_file = log_file
        self.auto_shutdown = auto_shutdown

        self.binary_path = find_ollama_binary(binary_path)
        self.process: Optional[subprocess.Popen] = None
        self._owned_process = False

        if self.auto_shutdown:
            atexit.register(self.stop)

    @property
    def native_base_url(self) -> str:
        """Returns the base native URL, e.g. http://127.0.0.1:11434"""
        return f"http://{self.host}:{self.port}"

    @property
    def openai_base_url(self) -> str:
        """Returns the OpenAI-compatible base URL, e.g. http://127.0.0.1:11434/v1"""
        return f"{self.native_base_url}/v1"

    def is_running(self) -> bool:
        """
        Checks if the Ollama server is alive and responding to health requests.
        """
        url = f"{self.native_base_url}/api/version"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as response:
                return response.status == 200
        except Exception:
            return False

    def start(self, timeout: float = 30.0, wait_ready: bool = True) -> None:
        """
        Starts the Ollama server subprocess if not already running.
        """
        if self.is_running():
            logger.info("Ollama server is already running at %s", self.native_base_url)
            return

        if not self.binary_path:
            raise FileNotFoundError(
                "Ollama binary not found. Please install it using ./install_ollama_portable.sh "
                "or export OLLAMA_BIN=/path/to/ollama."
            )

        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{self.host}:{self.port}"
        if self.models_dir:
            os.makedirs(self.models_dir, exist_ok=True)
            env["OLLAMA_MODELS"] = os.path.abspath(self.models_dir)
        if self.keep_alive:
            env["OLLAMA_KEEP_ALIVE"] = self.keep_alive

        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        if self.log_file:
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            log_f = open(self.log_file, "a", encoding="utf-8")
            self._log_fh = log_f
            stdout_dest = log_f
            stderr_dest = log_f

        logger.info("Starting Ollama server (%s serve) at %s:%s...", self.binary_path, self.host, self.port)
        self.process = subprocess.Popen(
            [self.binary_path, "serve"],
            env=env,
            stdout=stdout_dest,
            stderr=stderr_dest,
            start_new_session=True,
        )
        self._owned_process = True

        if wait_ready:
            ready = self.wait_until_ready(timeout=timeout)
            if not ready:
                self.stop()
                raise TimeoutError(f"Ollama server failed to start within {timeout}s.")
            logger.info("Ollama server is ready at %s", self.native_base_url)

    def wait_until_ready(self, timeout: float = 30.0, interval: float = 0.5) -> bool:
        """
        Blocks until the server responds or timeout is reached.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.is_running():
                return True
            if self.process and self.process.poll() is not None:
                logger.error("Ollama server process terminated with code %s", self.process.returncode)
                return False
            time.sleep(interval)
        return False

    def stop(self, timeout: float = 5.0) -> None:
        """
        Gracefully stops the Ollama server process if it was started by this instance.
        """
        if not self._owned_process or not self.process:
            return

        if self.process.poll() is None:
            logger.info("Stopping local Ollama server (PID: %s)...", self.process.pid)
            try:
                self.process.terminate()
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Ollama process did not terminate gracefully; sending SIGKILL.")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                logger.warning("Error while stopping Ollama process: %s", e)

        self.process = None
        self._owned_process = False

        if hasattr(self, "_log_fh"):
            self._log_fh.close()

    def restart(self, timeout: float = 30.0) -> None:
        """
        Restarts the Ollama server.
        """
        self.stop()
        time.sleep(1.0)
        self.start(timeout=timeout, wait_ready=True)

    def __enter__(self) -> "OllamaServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def ensure_server_running(
    binary_path: Optional[str] = None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    models_dir: Optional[str] = None,
    timeout: float = 30.0,
) -> OllamaServer:
    """
    Ensures that an Ollama server is running at host:port.
    Returns the active OllamaServer instance.
    """
    server = OllamaServer(
        binary_path=binary_path,
        host=host,
        port=port,
        models_dir=models_dir,
    )
    if not server.is_running():
        server.start(timeout=timeout, wait_ready=True)
    return server

