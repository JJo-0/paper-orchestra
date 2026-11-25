"""Native model APIs for provider-qualified writer and reviewer names."""
import base64
import os

import requests


def external_model(name):
    return name.startswith(("openai/", "anthropic/", "codex/", "claude/", "agy/"))


def generate(name, contents, config):
    provider, model = name.split("/", 1)
    if provider in {"codex", "claude", "agy"}:
        from utils.subscription_cli import generate as cli_generate
        return cli_generate(provider, model, contents, config)
    blocks = []
    for part in contents:
        text = part if isinstance(part, str) else part.text
        if text is not None:
            blocks.append({"type": "text" if provider == "anthropic" else "input_text", "text": text})
            continue
        blob = part.inline_data
        mime = blob.mime_type
        pdf = mime == "application/pdf"
        if not pdf and not mime.startswith("image/"):
            raise ValueError(f"Unsupported manuscript attachment: {mime}")
        data = base64.b64encode(blob.data).decode("ascii")
        if provider == "anthropic":
            blocks.append({"type": "document" if pdf else "image", "source": {
                "type": "base64", "media_type": mime, "data": data}})
        else:
            url = f"data:{mime};base64,{data}"
            blocks.append({"type": "input_file", "filename": "paper.pdf", "file_data": url}
                          if pdf else {"type": "input_image", "image_url": url})
    messages = [{"role": "user", "content": blocks}]
    instruction = config.get("system_instruction")
    search = bool(config.get("tools"))
    if provider == "openai":
        from utils.openai_utils import get_openai_client
        options = {}
        if instruction:
            options["instructions"] = instruction
        if search:
            options["tools"] = [{"type": "web_search"}]
        reply = get_openai_client().responses.create(model=model, input=messages, **options)
        if reply.status != "completed":
            raise RuntimeError(f"OpenAI response status: {reply.status}")
        return reply.output_text
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is required for anthropic/ models")
    payload = {"model": model, "messages": messages, "max_tokens": int(os.getenv("ANTHROPIC_MAX_TOKENS", "16000"))}
    if instruction:
        payload["system"] = instruction
    if search:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    response = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": key, "anthropic-version": "2023-06-01"}, json=payload, timeout=(10, 600))
    response.raise_for_status()
    result = response.json()
    if result["stop_reason"] != "end_turn":
        raise RuntimeError(f"Claude response stopped: {result['stop_reason']}")
    return "\n".join(b["text"] for b in result["content"] if b["type"] == "text")
