from __future__ import annotations

import os
import queue
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from memory.config_manager import get_input_device
from core import audio_devices


RATE = 16000
CHANNELS = 1
BLOCK = 1024


def _base_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_whisper() -> str | None:
    candidates = [
        os.getenv("MARK_LIV_WHISPER_BIN", ""),
        str(_base_dir() / "whisper.cpp" / "build" / "bin" / "whisper-cli"),
        str(_base_dir() / "whisper.cpp" / "build" / "bin" / "whisper-cli.exe"),
        "whisper-cli",
        "whisper-cli.exe",
    ]
    for p in candidates:
        if p and (Path(p).exists() or shutil.which(p)):
            return p
    return None


def whisper_model() -> str:
    return os.getenv(
        "MARK_LIV_WHISPER_MODEL",
        str(_base_dir() / "models" / "ggml-tiny.bin"),
    )


def _wav(path: str, pcm: bytes) -> None:
    with wave.open(path, "wb") as f:
        f.setnchannels(CHANNELS)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes(pcm)


def transcribe(pcm: bytes) -> str:
    binary = _find_whisper()
    model = whisper_model()
    if not binary:
        raise RuntimeError(
            "whisper-cli não encontrado. Instale/build o whisper.cpp e coloque "
            "whisper-cli em whisper.cpp/build/bin."
        )
    if not Path(model).exists():
        raise RuntimeError(f"Modelo Whisper local não encontrado: {model}")

    with tempfile.TemporaryDirectory(prefix="markliv-stt-") as td:
        wav_path = str(Path(td) / "input.wav")
        _wav(wav_path, pcm)
        cmd = [
            binary, "-m", model, "-f", wav_path,
            "-nt", "-np", "-l", "auto",
            "-t", os.getenv("MARK_LIV_WHISPER_THREADS", "2"),
        ]
        p = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if p.returncode != 0:
            raise RuntimeError((p.stderr or p.stdout).strip()[-1000:])
        lines = []
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line or line.startswith("["):
                continue
            lines.append(line)
        return " ".join(lines).strip()


class LocalMic:
    """Offline microphone + simple energy VAD.

    No audio is uploaded. Audio exists only in RAM and a short temporary WAV
    consumed by the local whisper.cpp process.
    """

    def __init__(self, wake_enabled=False, wake_detector=None, awake=True):
        self.wake_enabled = wake_enabled
        self.wake_detector = wake_detector
        self.awake = awake
        self._q: queue.Queue[bytes] = queue.Queue(maxsize=128)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if self.wake_enabled and not self.awake:
            if self.wake_detector is not None:
                self.wake_detector.feed(indata)
            return
        try:
            self._q.put_nowait(indata.tobytes())
        except queue.Full:
            pass

    def open(self):
        name = get_input_device()
        dev = audio_devices.resolve(name, "input")
        self._stream = sd.InputStream(
            samplerate=RATE, channels=CHANNELS, dtype="int16",
            blocksize=BLOCK, device=dev, callback=self._callback,
        )
        self._stream.start()
        return name

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_segment(self, min_seconds=0.30, max_seconds=15.0,
                     silence_seconds=0.85, threshold=0.012) -> bytes | None:
        pre = []
        active = []
        started = False
        silence_blocks = 0
        max_blocks = int(max_seconds * RATE / BLOCK)
        silence_limit = max(1, int(silence_seconds * RATE / BLOCK))
        min_bytes = int(min_seconds * RATE * 2)

        while len(active) < max_blocks:
            data = self._q.get()
            x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            loud = rms >= threshold

            if not started:
                pre.append(data)
                if len(pre) > 8:
                    pre.pop(0)
                if loud:
                    started = True
                    active.extend(pre)
                    pre.clear()
            else:
                active.append(data)
                if loud:
                    silence_blocks = 0
                else:
                    silence_blocks += 1
                    if silence_blocks >= silence_limit and len(b"".join(active)) >= min_bytes:
                        break

        pcm = b"".join(active)
        return pcm if len(pcm) >= min_bytes else None
