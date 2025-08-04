#!/usr/bin/env python3
"""Utility helpers around the local Ollama server.

Keeping these helpers in a dedicated module ensures that *text* helper
functions stay dependency-free (no *requests*, *subprocess*, …) which in
turn keeps import times low for modules that only need simple string
utilities.
"""

from __future__ import annotations

import subprocess
import time
from typing import List

import requests


_OLLAMA_URL = "http://localhost:11434"
# Default list of models required by the pipeline.  Additional models can
# be supplied to :func:`ensure_required_models` at runtime if needed.
_REQUIRED_MODELS: List[str] = ["gemma3n:e2b", "gemma3n:e4b"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_server_up() -> bool:
    try:
        return requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5).status_code == 200
    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ensure_ollama_running() -> bool:
    """Ensure the Ollama daemon is reachable; attempt to start it otherwise."""

    if _is_server_up():
        print("✅ Ollama is already running")
        return True

    print("🚀 Starting Ollama server…")
    try:
        # Start the server detached from the current process.  We discard
        # stdout/stderr to keep the console output of the main program clean.
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("❌ Ollama binary not found. Install it from https://ollama.ai/download")
        return False

    # Give the daemon up to 30 seconds to start.
    for _ in range(30):
        if _is_server_up():
            print("✅ Ollama started successfully")
            return True
        time.sleep(1)

    print("❌ Ollama did not start within 30 seconds")
    return False


def ensure_required_models(models: List[str] | None = None) -> bool:
    """Ensure all *models* are available locally (pull them if necessary).

    Parameters
    ----------
    models:
        A custom list of model identifiers to verify.  When *None* the
        default set defined in :data:`_REQUIRED_MODELS` is used.
    """

    models = models or _REQUIRED_MODELS

    try:
        response = requests.get(f"{_OLLAMA_URL}/api/tags", timeout=5)
        available = [m["name"] for m in response.json().get("models", [])]
    except requests.RequestException:
        print("❌ Cannot reach Ollama to verify local models")
        return False

    missing = [m for m in models if m not in available]

    if not missing:
        print("✅ All required models are available")
        return True

    print("📥 Pulling missing models:", ", ".join(missing))
    for model in missing:
        print(f"   ↳ {model}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes per model should be plenty
            )
            if result.returncode != 0:
                print(f"   ❌ Failed: {result.stderr.strip()}")
                return False
        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout while pulling {model}")
            return False
    return True
