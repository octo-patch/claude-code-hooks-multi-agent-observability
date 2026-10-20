import io
import json
import os
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch


HOOKS_DIR = Path(__file__).parents[1] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from utils.llm import minimax, registry
from utils.summarizer import generate_event_summary


class CaptureHandler(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        self.__class__.requests.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            response = {"choices": [{"message": {"content": "OpenAI response"}}]}
        else:
            response = {"content": [{"type": "text", "text": "Anthropic response"}]}
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format_string, *args):
        return


class MiniMaxProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        CaptureHandler.requests = []
        cls.server = HTTPServer(("127.0.0.1", 0), CaptureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def test_registry_contains_target_models_and_endpoints(self):
        self.assertEqual(
            list(minimax.MODEL_REGISTRY), ["MiniMax-M3.1-Flash", "MiniMax-M3"]
        )
        self.assertIs(registry.PROVIDER_REGISTRY["minimax"]["models"], minimax.MODEL_REGISTRY)
        self.assertEqual(
            minimax.ENDPOINT_REGISTRY["global_en"]["anthropic_base_url"],
            "https://api.minimax.io/anthropic",
        )
        self.assertEqual(
            minimax.ENDPOINT_REGISTRY["cn_zh"]["anthropic_base_url"],
            "https://api.minimaxi.com/anthropic",
        )

    def test_openai_compatible_request_path(self):
        config = {
            "api_key": "test-key",
            "protocol": "openai",
            "model": "MiniMax-M3",
            "base_url": f"{self.base_url}/v1",
        }
        response = minimax.request_completion("hello", config=config)
        self.assertEqual(response, "OpenAI response")
        self.assertEqual(CaptureHandler.requests[-1][0], "/v1/chat/completions")
        self.assertEqual(CaptureHandler.requests[-1][1]["model"], "MiniMax-M3")

    def test_anthropic_compatible_request_path(self):
        config = {
            "api_key": "test-key",
            "protocol": "anthropic",
            "model": "MiniMax-M2.7",
            "base_url": f"{self.base_url}/anthropic",
        }
        response = minimax.request_completion("hello", config=config)
        self.assertEqual(response, "Anthropic response")
        self.assertEqual(CaptureHandler.requests[-1][0], "/anthropic/v1/messages")
        self.assertEqual(CaptureHandler.requests[-1][1]["model"], "MiniMax-M2.7")


class MiniMaxDefaultRequestTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {"LLM_PROVIDER": "minimax", "MINIMAX_API_KEY": "test-key"},
            clear=True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        dotenv = patch.object(minimax, "load_dotenv")
        dotenv.start()
        self.addCleanup(dotenv.stop)

    @staticmethod
    def response():
        return io.BytesIO(json.dumps({
            "content": [{"type": "thinking", "thinking": "internal"},
                        {"type": "text", "text": "Ready"}],
            "choices": [{"message": {"content": "Ready", "reasoning_content": "internal"}}],
        }).encode())

    def test_registry_route_serializes_defaults_and_explicit_models(self):
        models = [None, "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-model"]
        for region, host in [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")]:
            for protocol, path in [("openai", "/v1/chat/completions"),
                                   ("anthropic", "/anthropic/v1/messages")]:
                for model in models:
                    with self.subTest(region=region, protocol=protocol, model=model):
                        settings = {"MINIMAX_REGION": region, "MINIMAX_PROTOCOL": protocol}
                        if model is not None:
                            settings["MINIMAX_MODEL"] = model
                        with patch.dict(os.environ, settings), patch.object(
                            minimax.urllib.request, "urlopen", return_value=self.response()
                        ) as transport:
                            self.assertEqual(registry.prompt_llm("hello"), "Ready")
                        transport.assert_called_once()
                        request = transport.call_args[0][0]
                        self.assertEqual(request.full_url, "https://" + host + path)
                        self.assertEqual(request.get_method(), "POST")
                        self.assertEqual(transport.call_args[1], {"timeout": 30})
                        # No disabled thinking, none effort, or unsupported tier is injected.
                        self.assertEqual(json.loads(request.data), {
                            "model": model or "MiniMax-M3.1-Flash",
                            "messages": [{"role": "user", "content": "hello"}],
                            "max_tokens": 100,
                        })
                        if protocol == "openai":
                            self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                            self.assertIsNone(request.get_header("X-api-key"))
                        else:
                            self.assertEqual(request.get_header("X-api-key"), "test-key")
                            self.assertEqual(request.get_header("Anthropic-version"), "2023-06-01")
                            self.assertIsNone(request.get_header("Authorization"))

    def test_hook_entry_points_use_flash_by_default(self):
        entry_points = [
            minimax.generate_completion_message,
            minimax.generate_agent_name,
            lambda: generate_event_summary({"hook_event_type": "Stop", "payload": {}}),
        ]
        for entry_point in entry_points:
            with self.subTest(entry_point=entry_point), patch.object(
                minimax.urllib.request, "urlopen", return_value=self.response()
            ) as transport:
                self.assertEqual(entry_point(), "Ready")
                request = transport.call_args[0][0]
                self.assertEqual(request.full_url, "https://api.minimax.io/anthropic/v1/messages")
                self.assertEqual(json.loads(request.data)["model"], "MiniMax-M3.1-Flash")

    def test_multimodal_content_is_preserved_at_transport(self):
        content = [{"type": "text", "text": "Describe this"},
                   {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}]
        with patch.dict(os.environ, {"MINIMAX_PROTOCOL": "openai"}), patch.object(
            minimax.urllib.request, "urlopen", return_value=self.response()
        ) as transport:
            self.assertEqual(registry.prompt_llm(content), "Ready")
            payload = json.loads(transport.call_args[0][0].data)
            self.assertEqual(payload["messages"][0]["content"], content)

    def test_invalid_configuration_does_not_send(self):
        for settings in [{"MINIMAX_MODEL": " "}, {"MINIMAX_REGION": "invalid"},
                         {"MINIMAX_PROTOCOL": "invalid"}, {"MINIMAX_API_KEY": ""}]:
            with self.subTest(settings=settings), patch.dict(os.environ, settings), patch.object(
                minimax.urllib.request, "urlopen"
            ) as transport:
                self.assertIsNone(registry.prompt_llm("hello"))
                transport.assert_not_called()

    def test_flash_metadata(self):
        model = minimax.MODEL_REGISTRY["MiniMax-M3.1-Flash"]
        self.assertEqual(model["context_window"], 1000000)
        self.assertEqual(model["max_output_tokens"], 524288)
        self.assertEqual(model["input_modalities"], ["text", "image", "video"])
        self.assertEqual(model["thinking"], ["adaptive"])
        self.assertEqual(model["reasoning_efforts"], ["low", "medium", "high", "xhigh", "max"])
        self.assertEqual(model["pricing_usd_per_million_tokens"], {
            "input": 0.15, "output": 0.6, "cache_read": 0.015, "cache_write": None,
        })


if __name__ == "__main__":
    unittest.main()
