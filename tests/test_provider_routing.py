import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from google.genai import types

from utils.gemini_utils import call_gemini_with_contents
from utils.llm_backend_utils import call_llm_with_text_prompt


class ProviderRouting(unittest.TestCase):
    @patch("utils.openai_utils.get_openai_client")
    def test_openai_writer_with_search_and_pdf(self, client):
        create = client.return_value.responses.create
        create.return_value = SimpleNamespace(status="completed", output_text='{"ok":true}')
        parts = ["read", types.Part.from_bytes(data=b"pdf", mime_type="application/pdf")]
        response = call_gemini_with_contents(parts, "openai/gpt-test", max_retries=1,
                                              generation_configs={"tools": [types.Tool(google_search=types.GoogleSearch())]})
        self.assertTrue(response["parsed_response"]["ok"])
        args = create.call_args.kwargs
        self.assertEqual(args["model"], "gpt-test")
        self.assertEqual(args["tools"], [{"type": "web_search"}])
        self.assertEqual(args["input"][0]["content"][1]["type"], "input_file")

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-only"})
    @patch("utils.provider_text.requests.post")
    def test_claude_reviewer(self, post):
        post.return_value.json.return_value = {"stop_reason": "end_turn", "content": [
            {"type": "text", "text": '{"score": 7}'}]}
        result = call_llm_with_text_prompt("Review", model_name="anthropic/claude-test",
                                           generation_configs={"system_instruction": "Be precise"})
        self.assertEqual(result["parsed_response"]["score"], 7)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "claude-test")
        self.assertEqual(payload["system"], "Be precise")
