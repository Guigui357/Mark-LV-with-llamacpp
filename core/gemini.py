from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from core.local_ai import LocalAI, local_model

FAST="fast"
SMART="smart"
SEARCH="search"
LIVE="live"
DEFAULT_TIMEOUT_MS=120000

class _Response:
    def __init__(self,raw):
        self.raw=raw
        choices=raw.get("choices") or []
        choice=choices[0] if choices else {}
        msg=choice.get("message") or {}
        self.text=str(msg.get("content") or "").strip()
        self.candidates=choices
        self.usage_metadata=raw.get("usage")

def api_key(refresh=False):
    return ""

def client(timeout_ms=DEFAULT_TIMEOUT_MS,key=""):
    class Models:
        @staticmethod
        def generate_content(model=None,contents=None,config=None):
            return call(contents,SMART,config)
    return SimpleNamespace(models=Models())

def live_model():
    return local_model()

def note_live_failure(model,err):
    return False

def is_quota_error(error): return False
def is_gone_error(error): return False
def is_unavailable_error(error): return False

def _parts(parts):
    text=[]
    rich=[]
    for part in parts or []:
        if isinstance(part,str):
            text.append(part)
        elif isinstance(part,dict):
            if "text" in part: text.append(str(part["text"]))
            elif "inline_data" in part:
                d=part["inline_data"]
                rich.append({"type":"image_url","image_url":{"url":f"data:{d.get('mime_type','image/png')};base64,{d.get('data','')}"}})
    if rich:
        if text: rich.append({"type":"text","text":"\n".join(text)})
        return rich
    return "\n".join(text)

def _messages(contents):
    if isinstance(contents,str): return [{"role":"user","content":contents}]
    if isinstance(contents,dict): contents=[contents]
    if not isinstance(contents,list): return [{"role":"user","content":str(contents)}]
    out=[]
    for item in contents:
        if isinstance(item,str): out.append({"role":"user","content":item})
        elif isinstance(item,dict):
            if "parts" in item:
                out.append({"role":item.get("role","user"),"content":_parts(item.get("parts") or [])})
            else:
                out.append({"role":item.get("role","user"),"content":item.get("content","")})
    return out

def call(contents,tier=FAST,config=None,timeout_ms=DEFAULT_TIMEOUT_MS,key=""):
    limits={FAST:384,SMART:768,SEARCH:768,LIVE:768}
    raw=LocalAI().chat(_messages(contents),None,0.2,limits.get(tier,768))
    return _Response(raw)

def text(contents,tier=FAST,config=None,timeout_ms=DEFAULT_TIMEOUT_MS,key="",default=""):
    try: return call(contents,tier,config,timeout_ms,key).text or default
    except Exception as exc:
        print(f"[LocalAI] text failed: {exc}")
        return default

def as_json(contents,tier=FAST,config=None,timeout_ms=DEFAULT_TIMEOUT_MS,key="",default=None):
    raw=text(contents,tier,config,timeout_ms,key,"")
    if not raw: return default
    if "{" in raw and "}" in raw: raw=raw[raw.find("{"):raw.rfind("}")+1]
    elif "[" in raw and "]" in raw: raw=raw[raw.find("["):raw.rfind("]")+1]
    try: return json.loads(raw)
    except Exception as exc:
        print(f"[LocalAI] JSON parse failed: {exc}")
        return default
