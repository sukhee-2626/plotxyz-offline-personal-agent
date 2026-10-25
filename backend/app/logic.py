"""Original deterministic implementation for Offline Personal (plotxyz-offline-personal-agent)."""
from __future__ import annotations

import ast
import csv
import io
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict, deque


class LogicError(ValueError):
    """Raised when supplied input cannot be processed."""


def _text(payload, key, min_len=1, max_len=20000):
    value = payload.get(key)
    if not isinstance(value, str):
        raise LogicError(f"'{key}' must be a string")
    value = value.strip()
    if len(value) < min_len:
        raise LogicError(f"'{key}' is required")
    if len(value) > max_len:
        raise LogicError(f"'{key}' exceeds {max_len} characters")
    return value


def _num(payload, key, default=None, lo=None, hi=None):
    value = payload.get(key, default)
    if value is None or value == "":
        if default is None:
            raise LogicError(f"'{key}' is required")
        return default
    if isinstance(value, bool):
        raise LogicError(f"'{key}' must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise LogicError(f"'{key}' must be numeric")
    if lo is not None and number < lo:
        raise LogicError(f"'{key}' must be at least {lo}")
    if hi is not None and number > hi:
        raise LogicError(f"'{key}' must be at most {hi}")
    return number


def _json(payload, key, expected=None, max_len=200000):
    raw = payload.get(key)
    if isinstance(raw, (dict, list)):
        data = raw
    elif isinstance(raw, str):
        if len(raw) > max_len:
            raise LogicError(f"'{key}' exceeds {max_len} characters")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LogicError(f"'{key}' is not valid JSON: {exc.msg} at line {exc.lineno}")
    else:
        raise LogicError(f"'{key}' must be JSON")
    if expected is not None and not isinstance(data, expected):
        name = expected.__name__ if hasattr(expected, "__name__") else str(expected)
        raise LogicError(f"'{key}' must be a JSON {name}")
    return data


def _csv(payload, key, max_rows=5000):
    text = _text(payload, key, max_len=2000000)
    try:
        reader = csv.DictReader(io.StringIO(text))
        if not reader.fieldnames:
            raise LogicError(f"'{key}' must include a header row")
        columns = [str(c).strip() for c in reader.fieldnames if c and str(c).strip()]
        if not columns:
            raise LogicError(f"'{key}' must include named columns")
        rows = []
        for index, raw in enumerate(reader):
            if index >= max_rows:
                raise LogicError(f"'{key}' exceeds {max_rows} rows")
            rows.append({col: (raw.get(col) or "").strip() for col in columns})
    except csv.Error as exc:
        raise LogicError(f"'{key}' is not valid CSV: {exc}")
    if not rows:
        raise LogicError(f"'{key}' must include at least one data row")
    return columns, rows


STOPWORDS = {'will', 'they', 'was', 'your', 'than', 'that', 'on', 'be', 'then', 'do', 'our', 'with', 'he', 'is', 'about', 'had', 'were', 'as', 'may', 'we', 'their', 'are', 'did', 'at', 'she', 'a', 'by', 'the', 'if', 'might', 'it', 'into', 'have', 'from', 'should', 'not', 'over', 'can', 'for', 'and', 'an', 'you', 'does', 'to', 'this', 'could', 'would', 'under', 'of', 'no', 'its', 'in', 'i', 'or', 'has'}


def _tokens(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9_+-]{1,}", text.lower()) if w not in STOPWORDS]


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip(" -•\t") for p in parts if len(p.strip()) > 2]


def provider_status():
    """Report which optional model providers are configured (never exposes secrets)."""
    ollama = bool(os.getenv("OLLAMA_BASE_URL"))
    openai = bool(os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_MODEL"))
    return {"ollama_compatible": ollama, "openai_compatible": openai}


def _llm(messages, temperature=0.2):
    """Call an explicitly configured OpenAI-compatible endpoint; return None when unavailable."""
    base = os.getenv("OPENAI_BASE_URL", "").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "")
    if not base or not model:
        ollama = os.getenv("OLLAMA_BASE_URL", "").rstrip("/")
        if not ollama:
            return None
        base = f"{ollama}/v1"
        model = os.getenv("OLLAMA_MODEL", "")
    if not model:
        return None
    try:
        import httpx

        headers = {"Content-Type": "application/json"}
        key = os.getenv("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        with httpx.Client(timeout=20) as client:
            response = client.post(
                f"{base}/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages, "temperature": temperature},
            )
            response.raise_for_status()
            return str(response.json()["choices"][0]["message"]["content"]).strip()
    except Exception:
        return None


def _result(summary, result, warnings=None):
    return {"summary": summary, "result": result, "warnings": warnings or []}


def agent_offline_personal(payload):
    message = _text(payload, "message", max_len=8000)
    context = str(payload.get("context", "")).strip()
    use_provider = bool(payload.get("use_provider", False))
    warnings, mode = [], "deterministic-fallback"
    reply = None
    if use_provider:
        reply = _llm([
            {"role": "system", "content": "You are a concise offline personal assistant. Do not claim to have used tools."},
            {"role": "user", "content": message + (f"\n\nContext:\n{context}" if context else "")},
        ])
        if reply:
            mode = "configured-provider"
        else:
            warnings.append("No OpenAI-compatible or Ollama endpoint is configured (or the call failed); deterministic fallback used.")
    if reply is None:
        lowered = message.lower()
        if re.fullmatch(r"[0-9\s+\-*/().%]+", message) and re.search(r"[+\-*/%]", message):
            try:
                node = ast.parse(message, mode="eval")
                allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.USub, ast.UAdd)
                if any(not isinstance(item, allowed) for item in ast.walk(node)):
                    raise ValueError("unsupported expression")
                value = eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {})
                reply = f"**Result:** `{value}`\n\nCalculated locally; no external service was called."
            except Exception:
                reply = "I could not evaluate that expression safely. Try plain arithmetic such as `(12 + 8) * 3`."
        elif lowered.startswith(("summar", "tl;dr")) and context:
            sentences = _sentences(context)
            picked = sentences[:3]
            reply = "**Local extractive summary**\n\n" + "\n".join(f"- {s}" for s in picked) + "\n\nOnly the supplied context was used."
        elif any(word in lowered for word in ("todo", "plan", "next step")):
            steps = [line.strip(" -") for line in context.splitlines() if line.strip()] if context else []
            if steps:
                reply = "**Plan from your notes**\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(steps[:8], 1))
            else:
                reply = "Here is a simple plan:\n\n1. State the outcome.\n2. List constraints.\n3. Take the smallest next action.\n4. Review the result."
        elif lowered.startswith(("hi", "hello", "hey")):
            reply = "Hello. I am the offline personal agent. I can calculate, summarize supplied context, and turn notes into a plan without a network connection."
        else:
            reply = (
                "**Offline response**\n\n"
                f"You said: {message}\n\n"
                "I can help with three local tasks: arithmetic, extractive summaries of pasted context, and turning notes into a plan. "
                "Configure `OPENAI_BASE_URL`/`OPENAI_MODEL` or `OLLAMA_BASE_URL` to enable optional model inference."
            )
    followups = ["Summarize the context I pasted", "Turn my notes into a plan", "Calculate (120 + 30) / 5"]
    return _result(
        f"Reply generated in {mode} mode.",
        {
            "markdown": reply,
            "metrics": {"mode": mode, "input_chars": len(message), "provider_configured": provider_status()},
            "items": followups,
        },
        warnings,
    )


run = agent_offline_personal
