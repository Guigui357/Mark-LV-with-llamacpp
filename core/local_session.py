from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from core.local_ai import LocalAI, to_openai_tools


@dataclass
class _FunctionCall:
    id: str
    name: str
    args: dict


class LocalSession:
    """Gemini-Live-shaped adapter over llama-server.

    It deliberately exposes only the tiny surface Mark-LIV's existing receive
    loop needs: send_client_content(), receive(), and send_tool_response().
    """

    def __init__(self, system_prompt: str, declarations: list[dict]):
        self.client = LocalAI()
        self.messages: list[dict] = [{"role": "system", "content": system_prompt}]
        # Keep declarations available locally, but do not send every schema on
        # every turn. On a 1.5B model, schemas can consume most of the context.
        self._declarations = [d for d in (declarations or []) if isinstance(d, dict)]
        self.tools: list[dict] = []
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._pending_tool_calls: list[dict] = []
        self.closed = False

    def _trim_context(self):
        if len(self.messages) <= 14:
            return
        tail = self.messages[-13:]
        while tail and tail[0].get("role") == "tool":
            tail.pop(0)
        self.messages = [self.messages[0]] + tail

    @staticmethod
    def _tool_text(declaration: dict) -> str:
        return " ".join([
            str(declaration.get("name") or ""),
            str(declaration.get("description") or ""),
        ]).lower()

    def _select_tools(self, user_text: str) -> list[dict]:
        """Select only tools relevant to the current user turn."""
        text = str(user_text or "").lower()
        if not text:
            return []

        action_words = (
            "abrir", "abre", "fechar", "fecha", "executar", "rodar", "iniciar",
            "parar", "criar", "apagar", "deletar", "mover", "renomear", "salvar",
            "lembrar", "lembre", "esquecer", "desfazer", "voltar", "cancelar",
            "ver", "mostrar", "olhar", "capturar", "tela", "câmera", "camera",
            "microfone", "wifi", "wi-fi", "internet", "volume", "brilho",
            "cpu", "ram", "memória", "memoria", "temperatura", "monitor",
            "arquivo", "pasta", "navegador", "browser", "pesquisar", "buscar",
            "notícia", "noticias", "news", "email", "mensagem", "download",
            "upload", "ligar", "desligar", "reiniciar", "shutdown",
        )
        if not any(word in text for word in action_words):
            return []

        aliases = {
            "memory": ("lembrar", "lembre", "memória", "memoria", "esquecer"),
            "screen": ("tela", "capturar", "mostrar", "olhar"),
            "camera": ("câmera", "camera"),
            "system": ("cpu", "ram", "memória", "memoria", "temperatura"),
            "wifi": ("wifi", "wi-fi", "internet"),
            "volume": ("volume",),
            "brightness": ("brilho",),
            "file": ("arquivo", "pasta", "salvar", "apagar", "deletar", "mover", "renomear"),
            "browser": ("navegador", "browser", "pesquisar", "buscar"),
            "web": ("pesquisar", "buscar", "notícia", "noticias", "news"),
            "email": ("email",),
            "message": ("mensagem",),
            "download": ("download",),
            "upload": ("upload",),
            "undo": ("desfazer", "voltar", "cancelar"),
            "open": ("abrir", "abre", "iniciar", "executar", "rodar"),
            "close": ("fechar", "fecha", "parar", "desligar"),
            "shutdown": ("reiniciar", "shutdown"),
        }

        scored = []
        for d in self._declarations:
            name = str(d.get("name") or "").lower()
            hay = self._tool_text(d)
            score = 0

            # Exact/partial tool-name matches are strong signals.
            for word in action_words:
                if word in name:
                    score += 4
                elif word in hay:
                    score += 1

            for alias, terms in aliases.items():
                if any(term in text for term in terms) and alias in hay:
                    score += 8

            if score:
                scored.append((score, d))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [d for _, d in scored[:10]]

    @staticmethod
    def _content(turns: Any) -> str | list:
        if isinstance(turns, str):
            return turns
        parts = (turns or {}).get("parts", []) if isinstance(turns, dict) else []
        text = []
        content = []
        for part in parts:
            if isinstance(part, str):
                text.append(part)
                continue
            if "text" in part:
                text.append(str(part["text"]))
            elif "inline_data" in part:
                d = part["inline_data"]
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": "data:%s;base64,%s" % (
                            d.get("mime_type", "image/png"),
                            d.get("data", ""),
                        )
                    },
                })
        if content:
            if text:
                content.append({"type": "text", "text": "\n".join(text)})
            return content
        return "\n".join(text)

    async def send_client_content(self, turns=None, turn_complete=True):
        if self.closed:
            return
        role = turns.get("role", "user") if isinstance(turns, dict) else "user"
        content = self._content(turns)
        if not content:
            return
        self.messages.append({"role": role, "content": content})
        if role == "user" and isinstance(content, str):
            self.tools = to_openai_tools(self._select_tools(content))
        self._trim_context()
        if turn_complete:
            await self._start_generation()

    async def _start_generation(self):
        if self._worker and not self._worker.done():
            return
        self._worker = asyncio.create_task(self._generate())

    async def _generate(self):
        try:
            result = await asyncio.to_thread(
                self.client.chat, self.messages, self.tools, 0.2, 512
            )
            choice = (result.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                self._pending_tool_calls = tool_calls
                calls = []
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    raw = fn.get("arguments", "{}")
                    if isinstance(raw, str):
                        try:
                            args = json.loads(raw) if raw.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    else:
                        args = raw if isinstance(raw, dict) else {}
                    calls.append(_FunctionCall(
                        id=str(tc.get("id") or ""),
                        name=str(fn.get("name") or ""),
                        args=args,
                    ))
                # Save the assistant tool-call message before the tool results.
                self.messages.append({
                    "role": "assistant",
                    "content": msg.get("content") or "",
                    "tool_calls": tool_calls,
                })
                await self._queue.put(SimpleNamespace(
                    data=None,
                    server_content=None,
                    tool_call=SimpleNamespace(function_calls=calls),
                ))
                return

            text = str(msg.get("content") or "").strip()
            self.messages.append({"role": "assistant", "content": text})
            sc = SimpleNamespace(
                output_transcription=SimpleNamespace(text=text),
                input_transcription=None,
                turn_complete=True,
            )
            await self._queue.put(SimpleNamespace(
                data=None,
                server_content=sc,
                tool_call=None,
            ))
        except Exception as e:
            text = f"Local AI error: {e}"
            sc = SimpleNamespace(
                output_transcription=SimpleNamespace(text=text),
                input_transcription=None,
                turn_complete=True,
            )
            await self._queue.put(SimpleNamespace(
                data=None, server_content=sc, tool_call=None
            ))

    async def receive(self):
        """Wait for and return the next local AI event."""
        if self.closed:
            return SimpleNamespace(data=None, server_content=None, tool_call=None)
        return await self._queue.get()

    async def send_tool_response(self, function_responses=None, defer_generation: bool = False):
        for fr in function_responses or []:
            fid = getattr(fr, "id", "")
            name = getattr(fr, "name", "")
            response = getattr(fr, "response", {}) or {}
            self.messages.append({
                "role": "tool",
                "tool_call_id": fid,
                "name": name,
                "content": json.dumps(response, ensure_ascii=False),
            })
        self._pending_tool_calls = []
        self._trim_context()
        if not defer_generation:
            await self._start_generation()

    async def generate(self):
        await self._start_generation()

    async def close(self):
        self.closed = True
        if self._worker and not self._worker.done():
            self._worker.cancel()

    def send_realtime_input(self, *args, **kwargs):
        raise RuntimeError("O modo de voz local usa STT/TTS local; áudio realtime Gemini não é usado.")
