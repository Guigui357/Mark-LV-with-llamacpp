from __future__ import annotations

import asyncio
import os
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace

from core.local_session import LocalSession
from core.local_ai import speak_local
from core import confirm as confirm_gate
from core import undo as undo_stack
from core.action_loader import discover_actions
from core.plugin_loader import discover_plugins
from actions.system_monitor import get_system_status, SystemMonitor
from actions.background_monitor import add_monitor, remove_monitor, list_monitors, check_all
from actions.proactive import ProactiveEngine
from memory.memory_manager import load_memory, update_memory, save_session_summary, search_memory

BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH = BASE_DIR / "core" / "prompt.txt"

TOOLS = [
    {"name":"system_status","description":"Return CPU, RAM, GPU, temperature, uptime and process metrics.","parameters":{"type":"OBJECT","properties":{}}},
    {"name":"save_memory","description":"Save a stable user fact, preference, project, relationship or plan.","parameters":{"type":"OBJECT","properties":{"category":{"type":"STRING"},"key":{"type":"STRING"},"value":{"type":"STRING"}},"required":["category","key","value"]}},
    {"name":"recall_memory","description":"Search long-term memory. Empty query lists stored facts.","parameters":{"type":"OBJECT","properties":{"query":{"type":"STRING"}}}},
    {"name":"undo","description":"Undo the last reversible change made by the assistant.","parameters":{"type":"OBJECT","properties":{"action":{"type":"STRING"}}}},
    {"name":"manage_monitor","description":"Add, remove or list background monitoring topics.","parameters":{"type":"OBJECT","properties":{"action":{"type":"STRING"},"topic":{"type":"STRING"}},"required":["action"]}},
    {"name":"shutdown_jarvis","description":"Shut down the assistant.","parameters":{"type":"OBJECT","properties":{}}},
]

def prompt():
    try: return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception: return "You are JARVIS, a concise local desktop assistant. Use tools instead of pretending."

class CLI:
    muted = False
    current_file = None

    def write_log(self, x): print(f"[LOG] {x}", flush=True)
    def set_state(self, x): print(f"[{x}]", flush=True)
    def set_audio_level(self, x): pass
    def show_content(self, title, content): print(f"\n--- {title} ---\n{content}\n---", flush=True)
    def show_confirm(self, title, detail):
        print(f"\n[CONFIRMAÇÃO] {title}\n{detail}\nDigite /confirm ou /cancel.", flush=True)
    def hide_confirm(self): pass
    def start_camera_stream(self): self.write_log("Camera stream não é exibido no CLI.")
    def stop_camera_stream(self): self.write_log("Camera stream parado.")
    def notify_phone_connected(self): self.write_log("Phone conectado.")
    def __getattr__(self, name):
        if name.startswith(("on_", "get_", "wake_")): return None
        raise AttributeError(name)

class MarkLIV:
    def __init__(self):
        self.ui = CLI()
        self.session = None
        self.stop = asyncio.Event()
        self.logs = []
        self.last_user = 0.0
        self.monitor = SystemMonitor()
        self.proactive = ProactiveEngine()

        names = {x["name"] for x in TOOLS}
        self.actions = discover_actions(
            actions_dir=BASE_DIR/"actions", reserved_names=names,
            logger=lambda x: print(f"[Actions] {x}", flush=True))
        names |= self.actions.names()
        self.plugins = discover_plugins(
            plugins_dir=BASE_DIR/"plugins", core_tool_names=names,
            logger=lambda x: print(f"[Plugins] {x}", flush=True),
            notify=lambda x: self.ui.write_log(f"SYS: {x}"))
        self.declarations = TOOLS + self.actions.get_tool_declarations() + self.plugins.get_tool_declarations()

    async def run(self):
        self.loop = asyncio.get_running_loop()
        confirm_gate.bind(self.ui.show_confirm, self.ui.hide_confirm, self.ui.write_log)

        from core.local_ai import LocalAI
        probe = LocalAI()
        if not await asyncio.to_thread(probe.health):
            print(f"[ERRO] llama serve não está disponível em {probe.base_url}", flush=True)
            return

        self.session = LocalSession(prompt(), self.declarations)
        self.last_user = self.loop.time()
        print("\n=== MARK-LIV CLI ===")
        print("llama.cpp local | sem GUI")
        print("Digite texto. /help /confirm /cancel /status /quit\n")

        try:
            await asyncio.gather(self.receive(), self.input_loop(), self.background())
        finally:
            if self.session: await self.session.close()
            await self.save_summary()

    async def input_loop(self):
        while self.session and not self.stop.is_set():
            try: text = (await asyncio.to_thread(input, "Você> ")).strip()
            except (EOFError, KeyboardInterrupt): self.stop.set(); break
            if not text: continue
            if text == "/help":
                print("/help /confirm /cancel /status /quit", flush=True); continue
            if text == "/confirm": confirm_gate.resolve(True); continue
            if text == "/cancel": confirm_gate.resolve(False); continue
            if text == "/status":
                print("llama serve: online" if await asyncio.to_thread(LocalAI_health) else "llama serve: offline"); continue
            if text == "/quit": self.stop.set(); break
            self.last_user = self.loop.time()
            self.logs.append("User: "+text)
            await self.session.send_client_content(turns={"role":"user","parts":[{"text":text}]}, turn_complete=True)

    async def receive(self):
        while self.session and not self.session.closed and not self.stop.is_set():
            try: response = await self.session.receive()
            except asyncio.CancelledError: raise
            except Exception as e:
                print(f"[ERRO] receive: {e}", flush=True); await asyncio.sleep(.2); continue

            tc = getattr(response, "tool_call", None)
            if tc:
                results = []
                for fc in getattr(tc, "function_calls", []) or []:
                    try: results.append(await self.tool(fc))
                    except Exception as e:
                        traceback.print_exc()
                        results.append(SimpleNamespace(id=getattr(fc,"id",""), name=getattr(fc,"name",""), response={"result":str(e)}))
                if results: await self.session.send_tool_response(results)
                continue

            sc = getattr(response, "server_content", None)
            if sc is None: continue
            out = getattr(sc, "output_transcription", None)
            text = " ".join(str(getattr(out,"text","") or "").split())
            if text:
                self.logs.append("JARVIS: "+text)
                print(f"\nJARVIS> {text}", flush=True)
                pbin = os.getenv("MARK_LIV_PIPER_BIN") or shutil.which("piper")
                pmodel = os.getenv("MARK_LIV_PIPER_MODEL")
                piper = bool(pbin and pmodel and Path(pmodel).exists())
                print("[TTS: Piper]" if piper else "[TTS: fallback]", flush=True)
                try:
                    if not os.getenv("MARK_LIV_NO_TTS"): await asyncio.to_thread(speak_local, text)
                except Exception as e: print(f"[ERRO] TTS: {e}", flush=True)

    async def tool(self, fc):
        name, args = fc.name, dict(fc.args or {})
        print(f"[TOOL] {name} {args}", flush=True)
        result = "Done."
        try:
            if name == "save_memory":
                update_memory({args.get("category","notes"):{args.get("key",""):{"value":args.get("value","")}}})
                result = "ok"
            elif name == "recall_memory":
                result = search_memory(args.get("query",""), limit=8)
            elif name == "undo":
                if str(args.get("action","")).lower()=="list":
                    h=undo_stack.history(); result="\n".join(f"{i+1}. {x}" for i,x in enumerate(h)) or "Nothing to undo."
                else: result=await asyncio.to_thread(undo_stack.undo_last)
            elif name == "system_status": result=str(await asyncio.to_thread(get_system_status))
            elif name == "manage_monitor":
                a=str(args.get("action","")).lower(); t=str(args.get("topic","")).strip()
                if a=="add" and t: result=await asyncio.to_thread(add_monitor,t)
                elif a=="remove" and t: result=await asyncio.to_thread(remove_monitor,t)
                elif a=="list": result="Monitoring: "+", ".join(await asyncio.to_thread(list_monitors))
                else: result="Specify add/remove/list and a topic."
            elif name == "shutdown_jarvis":
                result="Shutdown requested."; self.stop.set()
            elif self.actions.has(name):
                ctx={"player":self.ui,"speak":self.speak,"response":None,"session_memory":None}
                result=await asyncio.to_thread(self.actions.run,name,args,ctx) or "Done."
            elif self.plugins.has(name):
                result=await asyncio.to_thread(self.plugins.run,name,args,player=self.ui,session_memory=None) or "Done."
            else: result=f"Unknown tool: {name}"
        except Exception as e:
            traceback.print_exc(); result=f"Tool '{name}' failed: {e}"
        print(f"[TOOL RESULT] {name}: {str(result)[:160]}", flush=True)
        s=self.actions.scheduling(name) or self.plugins.scheduling(name)
        return SimpleNamespace(id=fc.id,name=name,response={"result":result},**({"scheduling":s} if s else {}))

    async def speak(self, text):
        if self.session:
            await self.session.send_client_content(turns={"role":"user","parts":[{"text":text}]}, turn_complete=True)

    async def background(self):
        while self.session and not self.stop.is_set():
            await asyncio.sleep(10)
            try:
                alert=await asyncio.to_thread(self.monitor.check)
                if alert and self.session:
                    await self.session.send_client_content(turns={"role":"user","parts":[{"text":alert}]},turn_complete=True)
            except Exception as e: print(f"[Monitor] {e}", flush=True)

    async def save_summary(self):
        if len(self.logs)<3: return
        mem=load_memory(); lang=mem.get("identity",{}).get("language",{})
        lang=(lang.get("value","") if isinstance(lang,dict) else str(lang)).strip() or "English"
        try:
            from core.local_ai import LocalAI
            r=await asyncio.to_thread(LocalAI().chat,[{"role":"system","content":"Summarize concisely. Output only the summary."},{"role":"user","content":f"Summarize in 1-2 sentences in {lang}:\n"+"\n".join(self.logs[-40:])}])
            c=r.get("choices") or []; s=((c[0].get("message") or {}).get("content") or "").strip() if c else ""
            if s: save_session_summary(s,lang)
        except Exception as e: print(f"[Memory] {e}",flush=True)

def LocalAI_health():
    try:
        from core.local_ai import LocalAI
        return LocalAI().health()
    except Exception: return False

def main():
    asyncio.run(MarkLIV().run())

if __name__=="__main__": main()
