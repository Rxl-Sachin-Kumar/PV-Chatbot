from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Generator, Iterator

import boto3
from botocore.exceptions import ClientError

from src.retrieval.retriever import retrieve
from src.utils.logger import get_logger

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS = 10
SUMMARY_THRESHOLD = 8
MAX_TOKENS        = 1024
MAX_RETRIES       = 3
BASE_BACKOFF      = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# MODELS — ordered by preference, latest non-deprecated first
# Nova models are pre-enabled in all AWS accounts (no access request needed)
# ─────────────────────────────────────────────────────────────────────────────

# Model families
NOVA_MODELS = [
    "amazon.nova-micro-v1:0",    # best quality, pre-enabled
    "amazon.nova-lite-v1:0",   # faster, cheaper
    "amazon.nova-pro-v1:0",  # fastest, text-only
]

CLAUDE_MODELS = [
    "anthropic.claude-3-5-sonnet-20241022-v2:0",  # best Claude, needs access grant
    "anthropic.claude-3-5-haiku-20241022-v1:0",   # fast Claude, needs access grant
    "anthropic.claude-3-sonnet-20240229-v1:0",    # older Claude fallback
]

# Full fallback chain: try Nova first (always available), then Claude if granted
FALLBACK_CHAIN = NOVA_MODELS + CLAUDE_MODELS

RETRYABLE_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "InternalServerException",
}

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are PharmaCog, an expert Pharmacovigilance (PV) AI assistant.

Your expertise covers:
- Adverse Drug Reactions (ADRs) and adverse events
- FAERS (FDA Adverse Event Reporting System) data interpretation
- Signal detection: disproportionality analysis (PRR, ROR, EBGM)
- Regulatory frameworks: ICH E2A/E2B/E2C, EU GVP modules, FDA regulations
- Causality assessment: WHO-UMC scale, Naranjo algorithm
- Risk management plans (RMP/REMS), post-marketing surveillance

Rules:
1. Base your answer ONLY on the provided CONTEXT chunks.
2. Always cite sources inline:
   - Books: [Book: <filename>, Page <page>]
   - FAERS:  [FAERS: Drug=<drug>, Reaction=<reaction>]
3. FAERS data = reported events, NOT confirmed causality. State this clearly.
4. Never fabricate statistics, dates, or regulatory decisions.
5. For clinical decisions, recommend consulting a healthcare professional.
6. If context is insufficient, say so explicitly — do not hallucinate."""

# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    DRUG_SAFETY   = "drug_safety"
    SIGNAL        = "signal_detection"
    REGULATORY    = "regulatory"
    GENERAL_PV    = "general_pv"
    UNKNOWN       = "unknown"

_INTENT_MAP = {
    QueryIntent.DRUG_SAFETY:  ["side effect", "adverse", "reaction", "toxicity", "adr", "harm", "risk"],
    QueryIntent.SIGNAL:       ["signal", "disproportionality", "prr", "ror", "ebgm", "detect", "mining"],
    QueryIntent.REGULATORY:   ["ich", "fda", "ema", "guideline", "regulation", "psur", "rmp", "rems"],
    QueryIntent.GENERAL_PV:   ["pharmacovigilance", "causality", "naranjo", "who-umc", "seriousness"],
}

def classify_intent(query: str) -> QueryIntent:
    lower = query.lower()
    scores = {intent: sum(1 for kw in kws if kw in lower)
              for intent, kws in _INTENT_MAP.items()}
    best = max(scores, key=lambda i: scores[i])
    return best if scores[best] > 0 else QueryIntent.UNKNOWN

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatResponse:
    answer: str
    sources: list
    intent: QueryIntent
    model_used: str
    latency_ms: float

# ─────────────────────────────────────────────────────────────────────────────
# BEDROCK CLIENT (singleton)
# ─────────────────────────────────────────────────────────────────────────────
_CLIENT = None

def get_client():
    global _CLIENT
    if _CLIENT is None:
        region = os.getenv("AWS_REGION", "us-east-1")

        # ✅ Use AWS profile (SSO)
        profile = os.getenv("AWS_PROFILE", "MLResearch-Team-533267396458")

        log.info("Initialising Bedrock client | region=%s | profile=%s", region, profile)

        session = boto3.Session(profile_name=profile)

        _CLIENT = session.client(
            "bedrock-runtime",
            region_name=region
        )

    return _CLIENT

# ─────────────────────────────────────────────────────────────────────────────
# REQUEST BODY BUILDER — correct format per model family
# ─────────────────────────────────────────────────────────────────────────────

def build_body(model_id: str, messages: list[dict], system: str) -> dict:
    mid = model_id.lower()

    # Amazon Nova
    if "nova" in mid or "amazon" in mid:
        return {
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}]}
                for m in messages
            ],
            "system": [{"text": system}],
            "inferenceConfig": {
                "maxTokens": MAX_TOKENS,
                "temperature": 0.2,
                "topP": 0.9,
            },
        }

    # Anthropic Claude (Bedrock)
    if "anthropic" in mid:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
        }

    # Generic fallback — try Nova format
    return {
        "messages": [
            {"role": m["role"], "content": [{"text": m["content"]}]}
            for m in messages
        ],
        "system": [{"text": system}],
        "inferenceConfig": {"maxTokens": MAX_TOKENS, "temperature": 0.2},
    }

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE EXTRACTOR — correct field per model family
# ─────────────────────────────────────────────────────────────────────────────

def extract_text(result: dict, model_id: str) -> str:
    mid = model_id.lower()

    # Amazon Nova
    if "nova" in mid or "amazon" in mid:
        try:
            return result["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError):
            pass

    # Anthropic Claude
    if "anthropic" in mid:
        try:
            return result["content"][0]["text"]
        except (KeyError, IndexError):
            pass

    # Generic fallback — try common keys
    for key in ("output", "content", "generation", "outputText", "results"):
        val = result.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            inner = val.get("message", {}).get("content", [{}])
            if inner and isinstance(inner[0], dict):
                return inner[0].get("text", str(val))
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, dict):
                return first.get("text", str(first))
            if isinstance(first, str):
                return first

    log.error("Could not extract text from response: %s", list(result.keys()))
    return str(result)

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING EVENT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_stream(event_stream, model_id: str) -> Generator[str, None, None]:
    mid = model_id.lower()
    try:
        for event in event_stream:
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = json.loads(chunk["bytes"].decode("utf-8"))

            # Amazon Nova streaming
            if "nova" in mid or "amazon" in mid:
                delta = data.get("contentBlockDelta", {}).get("delta", {})
                if "text" in delta:
                    yield delta["text"]

            # Anthropic Claude streaming
            elif "anthropic" in mid:
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

            # Generic
            else:
                for key in ("text", "generation", "outputText"):
                    if key in data:
                        yield data[key]
                        break

    except Exception as exc:
        log.error("Stream parse error: %s", exc)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER — rich metadata for LLM citation
# ─────────────────────────────────────────────────────────────────────────────

def build_context(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        src      = s.get("source", "unknown")
        drug     = s.get("drug", "")
        reaction = s.get("reaction", "")
        file_    = s.get("file", "")
        page     = s.get("page", "")
        score    = s.get("score", 0.0)

        if src == "book":
            cite = f"Book: {file_}, Page {page}" if page else f"Book: {file_}"
            header = f"[SOURCE {i}] TYPE=BOOK | CITE_AS=[{cite}] | Relevance={score:.3f}"
        else:
            cite = f"FAERS: Drug={drug}, Reaction={reaction}"
            header = f"[SOURCE {i}] TYPE=FAERS | CITE_AS=[{cite}] | Drug={drug} | Reaction={reaction} | Relevance={score:.3f}"

        parts.append(f"--- {header} ---\n{s['text']}\n")
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# CORE BEDROCK CALL — non-streaming with retry + fallback
# ─────────────────────────────────────────────────────────────────────────────

def call_bedrock(
    messages: list[dict],
    system: str = SYSTEM_PROMPT,
    model_chain: list[str] = FALLBACK_CHAIN,
) -> tuple[str, str]:
    client = get_client()

    for model in model_chain:
        log.info("Trying model: %s", model)
        body = json.dumps(build_body(model, messages, system))

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.invoke_model(
                    modelId=model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                result = json.loads(resp["body"].read())
                text = extract_text(result, model)
                log.info("Success: %s | attempt %d", model, attempt)
                return text, model

            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                msg  = exc.response["Error"]["Message"]

                if code in RETRYABLE_CODES:
                    wait = BASE_BACKOFF * (2 ** (attempt - 1))
                    log.warning("%s on %s attempt %d/%d — retry in %.1fs: %s",
                                code, model, attempt, MAX_RETRIES, wait, msg)
                    time.sleep(wait)
                    continue

                elif code in ("ValidationException", "AccessDeniedException",
                              "ResourceNotFoundException"):
                    log.warning("Model %s not available (%s) — next fallback", model, code)
                    break   # try next model

                else:
                    log.error("Fatal Bedrock error [%s]: %s", code, msg)
                    raise

            except Exception as exc:
                wait = BASE_BACKOFF * attempt
                log.warning("Attempt %d/%d failed (%s) — retry in %.1fs",
                            attempt, MAX_RETRIES, exc, wait)
                if attempt == MAX_RETRIES:
                    break
                time.sleep(wait)

    raise RuntimeError(
        "All Bedrock models failed. "
        "Check: aws configure | Bedrock model access | AWS_REGION in .env"
    )

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING BEDROCK CALL
# ─────────────────────────────────────────────────────────────────────────────

def call_bedrock_stream(
    messages: list[dict],
    system: str = SYSTEM_PROMPT,
    model_chain: list[str] = FALLBACK_CHAIN,
) -> tuple[Iterator[str], str]:
    client = get_client()

    for model in model_chain:
        log.info("Trying stream: %s", model)
        body = json.dumps(build_body(model, messages, system))

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.invoke_model_with_response_stream(
                    modelId=model,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )
                log.info("Streaming started: %s", model)
                return parse_stream(resp["body"], model), model

            except ClientError as exc:
                code = exc.response["Error"]["Code"]

                if code in RETRYABLE_CODES:
                    wait = BASE_BACKOFF * (2 ** (attempt - 1))
                    log.warning("%s — retry in %.1fs", code, wait)
                    time.sleep(wait)
                    continue

                elif code in ("ValidationException", "AccessDeniedException",
                              "ResourceNotFoundException"):
                    log.warning("Model %s not available (%s) — next fallback", model, code)
                    break

                else:
                    raise

            except Exception as exc:
                wait = BASE_BACKOFF * attempt
                log.warning("Stream attempt %d/%d failed (%s) — retry in %.1fs",
                            attempt, MAX_RETRIES, exc, wait)
                if attempt == MAX_RETRIES:
                    break
                time.sleep(wait)

    # If streaming fails entirely, fall back to non-streaming
    log.warning("Streaming failed — falling back to non-streaming")
    text, model_used = call_bedrock(messages, system, model_chain)
    return iter([text]), model_used

# ─────────────────────────────────────────────────────────────────────────────
# MAIN CHATBOT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class PVChatbot:
    """
    Stateful multi-turn Pharmacovigilance chatbot.
    Primary: Amazon Nova Pro (pre-enabled, no access request needed)
    Fallback: Nova Lite → Nova Micro → Claude 3.5 Sonnet → Claude Haiku

    Usage:
        bot = PVChatbot()
        response = bot.chat("What are the side effects of aspirin?")
        token_iter, meta = bot.stream_chat("Tell me more about GI bleeding")
        bot.reset()
    """

    def __init__(self, top_k: int = 6) -> None:
        self.top_k      = top_k
        self.history: list[Message] = []
        self.session_id = str(uuid.uuid4())
        self.turn       = 0
        log.info("PVChatbot ready | session=%s", self.session_id[:8])

    # ── Build messages payload ─────────────────────────────────────────────

    def _build_messages(self, query: str, context: str) -> list[dict]:
        messages = []

        # Include recent history (sliding window)
        recent = self.history[-(MAX_HISTORY_TURNS * 2):]
        for m in recent:
            messages.append({"role": m.role, "content": m.content})

        # Current query with RAG context
        messages.append({
            "role": "user",
            "content": (
                f"CONTEXT FROM KNOWLEDGE BASE:\n{context}\n"
                f"{'─' * 50}\n"
                f"QUESTION: {query}\n\n"
                f"Answer based on the context above. Cite sources inline."
            ),
        })
        return messages

    def _update_history(self, query: str, answer: str, intent: QueryIntent) -> None:
        self.history.append(Message("user", query))
        self.history.append(Message("assistant", answer))
        if len(self.history) > MAX_HISTORY_TURNS * 2:
            self.history = self.history[-(MAX_HISTORY_TURNS * 2):]

    # ── Non-streaming chat ─────────────────────────────────────────────────

    def chat(self, query: str) -> ChatResponse:
        t0 = time.perf_counter()
        self.turn += 1
        log.info("[Turn %d] %s", self.turn, query[:80])

        intent  = classify_intent(query)
        sources = retrieve(query, top_k=self.top_k, boost_faers=True)
        context = build_context(sources)

        messages = self._build_messages(query, context)
        answer, model_used = call_bedrock(messages)

        self._update_history(query, answer, intent)

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info("[Turn %d] Done %.0fms | model=%s", self.turn, latency_ms, model_used)

        return ChatResponse(
            answer=answer,
            sources=sources,
            intent=intent,
            model_used=model_used,
            latency_ms=latency_ms,
        )

    # ── Streaming chat ─────────────────────────────────────────────────────

    def stream_chat(self, query: str) -> tuple[Iterator[str], ChatResponse]:
        t0 = time.perf_counter()
        self.turn += 1
        log.info("[Turn %d | stream] %s", self.turn, query[:80])

        intent  = classify_intent(query)
        sources = retrieve(query, top_k=self.top_k, boost_faers=True)
        context = build_context(sources)
        messages = self._build_messages(query, context)

        raw_iter, model_used = call_bedrock_stream(messages)

        # Wrap iterator to capture full text and update history on completion
        buffer: list[str] = []

        def _collecting() -> Generator[str, None, None]:
            for token in raw_iter:
                buffer.append(token)
                yield token
            full = "".join(buffer)
            self._update_history(query, full, intent)
            log.info("[Turn %d] Stream done %.0fms | model=%s",
                     self.turn, (time.perf_counter() - t0) * 1000, model_used)

        meta = ChatResponse(
            answer="",          # filled after stream consumed
            sources=sources,
            intent=intent,
            model_used=model_used,
            latency_ms=0.0,
        )
        return _collecting(), meta

    # ── Session management ─────────────────────────────────────────────────

    def reset(self) -> None:
        old = self.session_id
        self.history    = []
        self.turn       = 0
        self.session_id = str(uuid.uuid4())
        log.info("Session reset | old=%s | new=%s", old[:8], self.session_id[:8])