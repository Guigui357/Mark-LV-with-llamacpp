from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import requests


def _base_dir() -> Path:
    if getattr(__import__("sys"), "frozen", False):
        return Path(__import__("sys").executable).parent
    return Path(__file__).resolve().parents[1]


BASE_DIR = _base_dir()
CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
DEFAULT_PIPER_MODEL = BASE_DIR / "pt_BR-faber-medium.onnx"
_piper_voice = None
_piper_lock = threading.Lock()


def load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def local_url() -> str:
    cfg = load_config()
    return str(cfg.get("local_ai_url") or os.getenv("MARK_LIV_LLM_URL") or "http://127.0.0.1:8080/v1").rstrip("/")


def local_model() -> str:
    cfg = load_config()
    return str(cfg.get("local_ai_model") or os.getenv("MARK_LIV_LLM_MODEL") or "mark-liv")


def _schema(v: Any) -> Any:
    if isinstance(v, list):
        return [_schema(x) for x in v]
    if not isinstance(v, dict):
        return v
    type_map = {
        "OBJECT": "object", "STRING": "string", "INTEGER": "integer",
        "NUMBER": "number", "BOOLEAN": "boolean", "ARRAY": "array",
        "NULL": "null",
    }
    out = {}
    for k, value in v.items():
        if k == "type" and isinstance(value, str):
            out[k] = type_map.get(value.upper(), value.lower())
        else:
            out[k] = _schema(value)
    return out


def to_openai_tools(declarations: list[dict] | None) -> list[dict]:
    result = []
    for d in declarations or []:
        if not isinstance(d, dict) or not d.get("name"):
            continue
        params = _schema(d.get("parameters") or {
            "type": "object", "properties": {}
        })
        if params.get("type") != "object":
            params = {"type": "object", "properties": {}}
        result.append({
            "type": "function",
            "function": {
                "name": d["name"],
                "description": str(d.get("description") or ""),
                "parameters": params,
            },
        })
    return result


class LocalAIError(RuntimeError):
    pass


class LocalAI:
    """Small HTTP client for llama.cpp's OpenAI-compatible local API."""

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or local_url()).rstrip("/")
        self.model = model or local_model()
        self.session = requests.Session()

    def health(self) -> bool:
        try:
            r = self.session.get(f"{self.base_url.rsplit('/v1', 1)[0]}/health", timeout=2)
            return r.ok
        except Exception:
            return False

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             temperature: float = 0.4, max_tokens: int = 1024) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        try:
            r = self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=600,
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            raise LocalAIError(
                f"Não foi possível falar com llama.cpp em {self.base_url}: {e}"
            ) from e


def _piper_model() -> Path | None:
    value = os.getenv("MARK_LIV_PIPER_MODEL", "").strip()
    path = Path(value).expanduser() if value else DEFAULT_PIPER_MODEL
    return path if path.exists() else None


def local_tts_backend() -> str:
    model = _piper_model()
    if model is None:
        return "fallback"
    try:
        from piper import PiperVoice
        return "Piper"
    except Exception:
        return "Piper" if (os.getenv("MARK_LIV_PIPER_BIN") or shutil.which("piper")) else "fallback"


def _play_wav(path: Path) -> None:
    import numpy as np
    import sounddevice as sd
    with wave.open(str(path), "rb") as f:
        data = f.readframes(f.getnframes())
        rate = f.getframerate()
        channels = f.getnchannels()
    pcm = np.frombuffer(data, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels)
    sd.play(pcm, rate, blocking=True)


def _speak_piper_python(text: str, model: Path) -> bool:
    global _piper_voice
    from piper import PiperVoice
    with _piper_lock:
        if _piper_voice is None:
            print(f"[TTS] Piper: loading {model}")
            _piper_voice = PiperVoice.load(str(model))
    with tempfile.TemporaryDirectory(prefix="marklv-tts-") as td:
        wav = Path(td) / "speech.wav"
        with wave.open(str(wav), "wb") as f:
            _piper_voice.synthesize_wav(text, f)
        _play_wav(wav)
    return True


def _speak_piper_binary(text: str, model: Path, binary: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="marklv-tts-") as td:
        wav = Path(td) / "speech.wav"
        p = subprocess.run(
            [binary, "--model", str(model), "--output_file", str(wav)],
            input=text, text=True, capture_output=True, check=False,
        )
        if p.returncode != 0 or not wav.exists():
            return False
        _play_wav(wav)
    return True


def _fallback_tts(text: str) -> bool:
    system = platform.system()
    try:
        if system == "Windows":
            escaped = text.replace("'", "''")
            script = (
                "Add-Type -AssemblyName System.Speech; "
                "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Speak('" + escaped + "');"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            return True
        if system == "Darwin" and shutil.which("say"):
            subprocess.run(["say", text], check=False)
            return True
        for cmd in ("espeak-ng", "espeak"):
            if shutil.which(cmd):
                subprocess.run([cmd, text], check=False)
                return True
    except Exception:
        pass
    return False


def speak_local(text: str) -> bool:
    text = " ".join(str(text or "").split()).strip()
    if not text:
        return False

    model = _piper_model()
    if model is not None:
        try:
            if _speak_piper_python(text, model):
                return True
        except Exception as exc:
            print(f"[TTS] Piper Python failed: {exc}")

        binary = os.getenv("MARK_LIV_PIPER_BIN") or shutil.which("piper")
        if binary:
            try:
                if _speak_piper_binary(text, model, binary):
                    return True
            except Exception as exc:
                print(f"[TTS] Piper binary failed: {exc}")

    return _fallback_tts(text)
