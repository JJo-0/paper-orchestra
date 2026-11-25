"""Run one manuscript turn using an already authenticated local CLI."""
import os
from pathlib import Path
import subprocess
import tempfile


def generate(provider, model, contents, config):
    with tempfile.TemporaryDirectory(prefix="paper-orchestra-") as workspace:
        root = Path(workspace)
        parts = [config["system_instruction"]] if config.get("system_instruction") else []
        images = []
        for index, part in enumerate(contents):
            text = part if isinstance(part, str) else part.text
            if text is not None:
                parts.append(text)
                continue
            blob = part.inline_data
            suffix = {"application/pdf": ".pdf", "image/png": ".png", "image/jpeg": ".jpg"}[blob.mime_type]
            file = root / f"attachment-{index}{suffix}"
            file.write_bytes(blob.data)
            parts.append(f"Read the supplied attachment {file.name}.")
            if suffix != ".pdf":
                images.extend(["--image", str(file)])
        search = bool(config.get("tools"))
        if search:
            parts.append("Search the web for primary sources and verify the references.")
        prompt = "\n\n".join(parts)
        select = [] if model == "default" else ["--model", model]
        env = os.environ.copy()
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            env.pop(key, None)
        if provider == "codex":
            output = root / "answer.txt"
            cmd = ["codex", *(["--search"] if search else []), "exec", "--ephemeral",
                   "--sandbox", "read-only", "--skip-git-repo-check", *select, *images, "-o", str(output), "-"]
        elif provider == "claude":
            cmd = ["claude", "-p", "--no-session-persistence", "--output-format", "text",
                   "--tools", "Read,WebSearch,WebFetch" if search else "Read", *select]
        else:
            # AGY print mode takes a prompt argument. Read the long manuscript from a local file.
            request = prompt
            if len(prompt.encode()) > 64000:
                (root / "request.txt").write_text(prompt)
                request = "Read request.txt in the working directory and answer it directly."
            cmd = ["agy", "--sandbox", "--mode", "plan", "--add-dir", workspace, *select,
                   "--print-timeout", "8m", "-p", request]
        run = subprocess.run(cmd, input=prompt if provider != "agy" else "", text=True,
                             capture_output=True, cwd=root, env=env, timeout=510)
        if run.returncode:
            raise RuntimeError(f"{provider} exited {run.returncode}: {(run.stderr or run.stdout)[-600:]}")
        return output.read_text().strip() if provider == "codex" else run.stdout.strip()
