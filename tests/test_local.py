"""
Unit tests for arc_cegis.local subpackage and Ollama integration.
"""

import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from arc_cegis import config, llm
from arc_cegis.local import (
    OllamaClient,
    OllamaServer,
    ensure_server_running,
    find_ollama_binary,
    get_local_openai_client,
    setup_local_ollama,
)


class TestOllamaBinaryDiscovery(unittest.TestCase):
    def test_custom_path_found(self):
        with patch("os.path.isfile", return_value=True), patch("os.access", return_value=True):
            binary = find_ollama_binary("/custom/path/to/ollama")
            self.assertEqual(binary, os.path.abspath("/custom/path/to/ollama"))

    def test_env_path_found(self):
        with patch.dict(os.environ, {"OLLAMA_BIN": "/env/path/to/ollama"}), \
             patch("os.path.isfile", side_effect=lambda p: p == "/env/path/to/ollama"), \
             patch("os.access", return_value=True):
            binary = find_ollama_binary()
            self.assertEqual(binary, os.path.abspath("/env/path/to/ollama"))

    def test_system_path_fallback(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("os.path.isfile", return_value=False), \
             patch("shutil.which", return_value="/usr/bin/ollama"):
            binary = find_ollama_binary()
            self.assertEqual(binary, "/usr/bin/ollama")


class TestOllamaServer(unittest.TestCase):
    def setUp(self):
        self.server = OllamaServer(
            binary_path="/fake/bin/ollama",
            host="127.0.0.1",
            port=11434,
            auto_shutdown=False,
        )

    def test_urls(self):
        self.assertEqual(self.server.native_base_url, "http://127.0.0.1:11434")
        self.assertEqual(self.server.openai_base_url, "http://127.0.0.1:11434/v1")

    @patch("urllib.request.urlopen")
    def test_is_running_true(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(self.server.is_running())

    @patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused"))
    def test_is_running_false(self, mock_urlopen):
        self.assertFalse(self.server.is_running())

    @patch("subprocess.Popen")
    def test_start_and_stop(self, mock_popen):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_popen.return_value = mock_proc

        with patch.object(self.server, "is_running", side_effect=[False, True]), \
             patch.object(self.server, "binary_path", "/fake/bin/ollama"):
            self.server.start(timeout=5.0, wait_ready=True)
            self.assertTrue(self.server._owned_process)
            self.assertEqual(self.server.process, mock_proc)

            self.server.stop()
            mock_proc.terminate.assert_called_once()
            self.assertIsNone(self.server.process)


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(host="127.0.0.1", port=11434)

    @patch("urllib.request.urlopen")
    def test_ping_and_version(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"version": "0.5.12"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        self.assertTrue(self.client.ping())
        self.assertEqual(self.client.get_version(), "0.5.12")

    @patch("urllib.request.urlopen")
    def test_list_models_and_availability(self, mock_urlopen):
        tags_payload = {
            "models": [
                {"name": "qwen2.5-coder:7b", "model": "qwen2.5-coder:7b"},
                {"name": "llama3.2:latest", "model": "llama3.2:latest"},
            ]
        }
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps(tags_payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        names = self.client.list_model_names()
        self.assertIn("qwen2.5-coder:7b", names)
        self.assertIn("llama3.2:latest", names)

        self.assertTrue(self.client.is_model_available("qwen2.5-coder:7b"))
        self.assertTrue(self.client.is_model_available("llama3.2"))
        self.assertFalse(self.client.is_model_available("mistral:7b"))

    @patch("urllib.request.urlopen")
    def test_pull_model(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = json.dumps({"status": "success"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        success = self.client.pull_model("qwen2.5-coder:7b")
        self.assertTrue(success)


class TestLocalIntegration(unittest.TestCase):
    def test_config_defaults(self):
        self.assertEqual(config.get_api_key("ollama"), "ollama")
        self.assertEqual(config.get_api_key("local"), "ollama")
        self.assertEqual(config.get_api_base_url("ollama"), "http://127.0.0.1:11434/v1")
        self.assertEqual(config.get_api_base_url("local"), "http://127.0.0.1:11434/v1")

    def test_get_local_openai_client(self):
        client = get_local_openai_client()
        self.assertIsNotNone(client)
        self.assertEqual(str(client.base_url), "http://127.0.0.1:11434/v1/")

    @patch("arc_cegis.local.client.OllamaClient.ping", return_value=True)
    @patch("arc_cegis.local.client.OllamaClient.is_model_available", return_value=True)
    def test_setup_local_ollama(self, mock_avail, mock_ping):
        server, client = setup_local_ollama(
            model="qwen2.5-coder:7b",
            auto_start=False,
            auto_pull=True,
            set_global_config=True,
        )
        self.assertEqual(config.LLM_PROVIDER, "ollama")
        self.assertEqual(config.MODEL_NAME, "qwen2.5-coder:7b")
        self.assertEqual(config.API_BASE_URL, "http://127.0.0.1:11434/v1")
        self.assertEqual(config.REQUEST_DELAY, 0.0)


if __name__ == "__main__":
    unittest.main()

