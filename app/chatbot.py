from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# app/chatbot.py — Production-grade stateful multi-turn PV chatbot.
# See README.md for full architecture documentation.


import json
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Generator, Iterator, Optional

import boto3
from botocore.exceptions import ClientError, EndpointResolutionError

from src.retrieval.retriever import retrieve
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── AWS Bedrock model config ──────────────────────────────────────────────────

class BedrockModel(str, Enum):
    # Claude models
    CLAUDE_35_SONNET  = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_35_HAIKU   = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_3_SONNET   = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_HAIKU    = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_INSTANT    = "anthropic.claude-instant-v1"
    CLAUDE_2_1        = "anthropic.claude-v2:1"
    # Amazon Nova (widely available, no special access needed)
    NOVA_PRO          = "amazon.nova-pro-v1:0"
    NOVA_LITE         = "amazon.nova-lite-v1:0"
    NOVA_MICRO        = "amazon.nova-micro-v1:0"
    # Meta Llama
    LLAMA3_70B        = "meta.llama3-70b-instruct-v1:0"
    LLAMA3_8B         = "meta.llama3-8b-instruct-v1:0"
    # Mistral
    MISTRAL_LARGE     = "mistral.mistral-large-2402-v1:0"
    MISTRAL_7B        = "mistral.mistral-7b-instruct-v0:2"

FALLBACK_CHAIN = [
    BedrockModel.CLAUDE_35_SONNET,
    BedrockModel.CLAUDE_35_HAIKU,
    BedrockModel.CLAUDE_3_SONNET,
    BedrockModel.CLAUDE_3_HAIKU,
    BedrockModel.CLAUDE_INSTANT,
    BedrockModel.CLAUDE_2_1,
    BedrockModel.NOVA_PRO,
    BedrockModel.NOVA_LITE,
    BedrockModel.NOVA_MICRO,
    BedrockModel.LLAMA3_70B,
    BedrockModel.LLAMA3_8B,
    BedrockModel.MISTRAL_LARGE,
    BedrockModel.MISTRAL_7B,
]

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES      = 4
BASE_BACKOFF     = 1.5    # seconds
MAX_BACKOFF      = 30.0
RETRYABLE_CODES  = {"ThrottlingException", "ServiceUnavailableException",
                    "ModelTimeoutException", "InternalServerException"}

# ── Conversation config ───────────────────────────────────────────────────────
MAX_HISTORY_TURNS   = 10    # keep last N user+assistant pairs in context
SUMMARY_THRESHOLD   = 8     # summarise when turns exceed this
MAX_CONTEXT_CHUNKS  = 6     # retrieved chunks per query
MAX_TOKENS_RESPONSE = 1024

# ── PV Domain System Prompt ───────────────────────────────────────────────────
PV_SYSTEM_PROMPT = """You are PharmaCog, an expert AI assistant specialising in Pharmacovigilance (PV) and Drug Safety.

Your knowledge covers:
- Adverse Drug Reactions (ADRs) and adverse events
- FAERS (FDA Adverse Event Reporting System) data and interpretation
- Signal detection and risk assessment
- Regulatory frameworks: ICH E2A/E2B/E2C/E2D/E2E, EU GVP modules, FDA regulations
- Causality assessment (WHO-UMC, Naranjo scale)
- Risk management plans (RMPs) and Risk Evaluation and Mitigation Strategies (REMS)
- Pharmacoepidemiology methods
- Drug-drug interactions with safety implications
- Post-marketing surveillance

Guidelines:
1. Always base answers on the provided CONTEXT chunks. If the context is insufficient, say so clearly.
2. When citing FAERS data, note it represents REPORTED events — not confirmed causal relationships.
3. For clinical decisions, always recommend consulting a qualified healthcare professional.
4. Use precise medical terminology appropriate for PV professionals.
5. When a drug is mentioned, proactively consider: indication, known ADR profile, reporting rates.
6. Flag any POTENTIAL SAFETY SIGNALS with appropriate uncertainty language.
7. Never fabricate adverse event statistics or regulatory dates.
8. Maintain conversation continuity — reference prior turns when relevant.

Response format:
- Lead with a direct answer
- Support with evidence from context
- ALWAYS cite sources inline using this exact format:
    • For books:  [Book: <filename>, Page <page_number>]
    • For FAERS:  [FAERS: Drug=<drug_name>, Reaction=<reaction>]
  Example: "Aspirin is associated with GI bleeding [Book: Pharmacovigilance.pdf, Page 42] and has been reported in FAERS [FAERS: Drug=ASPIRIN, Reaction=GASTROINTESTINAL HAEMORRHAGE]"
- If multiple sources support a claim, cite all of them
- End serious safety topics with a clinical disclaimer"""

# ── Intent classification ─────────────────────────────────────────────────────

class QueryIntent(str, Enum):
    DRUG_SAFETY     = "drug_safety"       # ADRs, side effects for a specific drug
    SIGNAL_DETECT   = "signal_detection"  # signal detection methods/findings
    REGULATORY      = "regulatory"        # guidelines, regulations, ICH, FDA
    EPIDEMIOLOGY    = "epidemiology"      # study design, RWE, pharmacoepi
    GENERAL_PV      = "general_pv"        # general pharmacovigilance concepts
    UNKNOWN         = "unknown"

_INTENT_KEYWORDS: dict[QueryIntent, list[str]] = {
    QueryIntent.DRUG_SAFETY: [
        "side effect", "adverse", "reaction", "toxicity", "safety", "adr",
        "harm", "risk", "dose", "overdose", "interaction", "contraindic",
    ],
    QueryIntent.SIGNAL_DETECT: [
        "signal", "disproportionality", "prr", "ror", "ebgm", "bcpnn",
        "detect", "mining", "spontaneous", "report",
    ],
    QueryIntent.REGULATORY: [
        "ich", "fda", "ema", "regulation", "guideline", "directive",
        "psur", "pbrer", "rmp", "rems", "label", "approval", "submission",
    ],
    QueryIntent.EPIDEMIOLOGY: [
        "cohort", "case-control", "incidence", "prevalence", "epidemiology",
        "real-world", "rwe", "database", "study design",
    ],
    QueryIntent.GENERAL_PV: [
        "pharmacovigilance", "causality", "naranjo", "who-umc", "expectedness",
        "seriousness", "listedness", "expedited", "periodic",
    ],
}


def classify_intent(query: str) -> QueryIntent:
    lower = query.lower()
    scores: dict[QueryIntent, int] = {intent: 0 for intent in QueryIntent}
    for intent, keywords in _INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                scores[intent] += 1
    best = max(scores, key=lambda i: scores[i])
    return best if scores[best] > 0 else QueryIntent.UNKNOWN


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Message:
    role: str          # "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    intent: Optional[QueryIntent] = None


@dataclass
class ChatResponse:
    answer: str
    sources: list[dict]
    intent: QueryIntent
    model_used: str
    session_id: str
    turn: int
    latency_ms: float
    retrieved_count: int


# ── Bedrock client (singleton) ────────────────────────────────────────────────

_BEDROCK_CLIENT = None

def _get_bedrock_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
        log.info("Initialising Bedrock client in region: %s", region)
        _BEDROCK_CLIENT = boto3.client(
            service_name="bedrock-runtime",
            region_name=region,
        )
    return _BEDROCK_CLIENT


# ── Response text extractor (handles different model schemas) ────────────────

def _extract_text(result: dict, model_id: str) -> str:
    """
    Extract response text from Bedrock response.
    Different model families return different response shapes.
    """
    mid = model_id.lower()

    # Claude — {"content": [{"type": "text", "text": "..."}]}
    if "anthropic" in mid:
        return result["content"][0]["text"]

    # Amazon Nova — {"output": {"message": {"content": [{"text": "..."}]}}}
    if "nova" in mid or "amazon" in mid:
        try:
            return result["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError):
            return result.get("outputText", str(result))

    # Meta Llama — {"generation": "..."}
    if "llama" in mid or "meta" in mid:
        return result.get("generation", str(result))

    # Mistral — {"outputs": [{"text": "..."}]}
    if "mistral" in mid:
        try:
            return result["outputs"][0]["text"]
        except (KeyError, IndexError):
            return str(result)

    # Generic fallback — try common keys
    for key in ("content", "generation", "outputText", "output", "results"):
        if key in result:
            val = result[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                first = val[0]
                if isinstance(first, dict):
                    return first.get("text", first.get("outputText", str(first)))
                return str(first)
            if isinstance(val, dict):
                return val.get("message", {}).get("content", [{}])[0].get("text", str(val))

    return str(result)


# ── Per-model request body builder ───────────────────────────────────────────

def _build_request_body(model_id: str, messages: list[dict], system_prompt: str) -> dict:
    """Build the correct request payload for each model family."""
    mid = model_id.lower()

    # ── Claude (Anthropic) ────────────────────────────────────
    if "anthropic" in mid:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": MAX_TOKENS_RESPONSE,
            "system": system_prompt,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
        }

    # ── Amazon Nova ───────────────────────────────────────────
    if "nova" in mid or "amazon" in mid:
        # Nova uses same Converse-style messages but different schema
        nova_messages = []
        for m in messages:
            nova_messages.append({
                "role": m["role"],
                "content": [{"text": m["content"]}],
            })
        return {
            "messages": nova_messages,
            "system": [{"text": system_prompt}],
            "inferenceConfig": {
                "maxTokens": MAX_TOKENS_RESPONSE,
                "temperature": 0.2,
                "topP": 0.9,
            },
        }


    # ── Meta Llama ────────────────────────────────────────────
    if "llama" in mid or "meta" in mid:
        prompt_parts = ["<|system|>\n" + system_prompt + "\n"]
        for m in messages:
            role = "user" if m["role"] == "user" else "assistant"
            prompt_parts.append("<|" + role + "|>\n" + m["content"] + "\n")
        prompt_parts.append("<|assistant|>\n")
        return {
            "prompt": "".join(prompt_parts),
            "max_gen_len": MAX_TOKENS_RESPONSE,
            "temperature": 0.2,
            "top_p": 0.9,
        }

    # ── Mistral ───────────────────────────────────────────────
    if "mistral" in mid:
        prompt_parts = ["<s>[INST] " + system_prompt + "\n\n"]
        for m in messages:
            if m["role"] == "user":
                prompt_parts.append(m["content"])
                prompt_parts.append(" [/INST] ")
            else:
                prompt_parts.append(m["content"])
                prompt_parts.append(" </s><s>[INST] ")
        return {
            "prompt": "".join(prompt_parts),
            "max_tokens": MAX_TOKENS_RESPONSE,
            "temperature": 0.2,
            "top_p": 0.9,
        }


    # ── Generic fallback (try Claude format) ──────────────────
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": MAX_TOKENS_RESPONSE,
        "system": system_prompt,
        "messages": messages,
        "temperature": 0.2,
    }


# ── Core LLM call with retry + fallback ──────────────────────────────────────

def _invoke_with_retry(
    messages: list[dict],
    system_prompt: str,
    model_chain: list[BedrockModel] = FALLBACK_CHAIN,
) -> tuple[str, str]:
    """
    Call Bedrock with retry + model fallback.

    Returns (response_text, model_id_used)
    """
    client = _get_bedrock_client()

    for model in model_chain:
        log.info("Trying model: %s", model.value)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                body = json.dumps(_build_request_body(model.value, messages, system_prompt))

                resp = client.invoke_model(
                    modelId=model.value,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                )

                result = json.loads(resp["body"].read())
                text = _extract_text(result, model.value)
                log.info("Success with model %s on attempt %d", model.value, attempt)
                return text, model.value

            except ClientError as exc:
                error_code = exc.response["Error"]["Code"]
                error_msg  = exc.response["Error"]["Message"]

                if error_code in RETRYABLE_CODES:
                    wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                    log.warning(
                        "%s on %s attempt %d/%d — retrying in %.1fs: %s",
                        error_code, model.value, attempt, MAX_RETRIES, wait, error_msg,
                    )
                    time.sleep(wait)
                    continue

                elif error_code in ("ValidationException", "AccessDeniedException",
                                    "ResourceNotFoundException"):
                    # Model not available in this region/account — try next model
                    log.warning("Model %s not available (%s) — trying fallback.", model.value, error_code)
                    break   # break retry loop, try next model

                else:
                    log.error("Non-retryable Bedrock error [%s]: %s", error_code, error_msg)
                    raise

            except (EndpointResolutionError, Exception) as exc:
                if attempt == MAX_RETRIES:
                    log.error("All retries exhausted for model %s: %s", model.value, exc)
                    break
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                log.warning("Attempt %d failed: %s — retrying in %.1fs", attempt, exc, wait)
                time.sleep(wait)

    raise RuntimeError(
        "All Bedrock models exhausted. Check AWS credentials, region, and model access."
    )




# ── Streaming LLM call with retry + fallback ─────────────────────────────────

def _stream_with_retry(
    messages: list[dict],
    system_prompt: str,
    model_chain: list[BedrockModel] = FALLBACK_CHAIN,
) -> tuple[Iterator[str], str]:
    """
    Stream tokens from Bedrock with retry + model fallback.

    Returns (token_iterator, model_id_used)

    Only Claude and Nova support streaming via invoke_model_with_response_stream.
    Llama/Mistral/Titan fall back to non-streaming invoke.
    """
    client = _get_bedrock_client()

    STREAMING_SUPPORTED = ("anthropic", "nova", "amazon")

    for model in model_chain:
        mid = model.value.lower()
        log.info("Trying model (stream): %s", model.value)
        supports_stream = any(p in mid for p in STREAMING_SUPPORTED)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                body = json.dumps(_build_request_body(model.value, messages, system_prompt))

                if supports_stream:
                    resp = client.invoke_model_with_response_stream(
                        modelId=model.value,
                        body=body,
                        contentType="application/json",
                        accept="application/json",
                    )
                    return _parse_stream(resp["body"], model.value), model.value
                else:
                    # Non-streaming fallback for Llama/Mistral
                    resp = client.invoke_model(
                        modelId=model.value,
                        body=body,
                        contentType="application/json",
                        accept="application/json",
                    )
                    result = json.loads(resp["body"].read())
                    text = _extract_text(result, model.value)
                    log.info("Non-streaming success: %s", model.value)
                    return iter([text]), model.value

            except ClientError as exc:
                error_code = exc.response["Error"]["Code"]
                error_msg  = exc.response["Error"]["Message"]

                if error_code in RETRYABLE_CODES:
                    wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                    log.warning("%s on %s attempt %d/%d — retrying in %.1fs",
                                error_code, model.value, attempt, MAX_RETRIES, wait)
                    time.sleep(wait)
                    continue

                elif error_code in ("ValidationException", "AccessDeniedException",
                                    "ResourceNotFoundException"):
                    log.warning("Model %s not available (%s) — trying fallback.",
                                model.value, error_code)
                    break

                else:
                    log.error("Non-retryable error [%s]: %s", error_code, error_msg)
                    raise

            except Exception as exc:
                if attempt == MAX_RETRIES:
                    log.error("All retries exhausted for %s: %s", model.value, exc)
                    break
                wait = min(BASE_BACKOFF * (2 ** (attempt - 1)), MAX_BACKOFF)
                log.warning("Attempt %d failed: %s — retrying in %.1fs", attempt, exc, wait)
                time.sleep(wait)

    raise RuntimeError(
        "All Bedrock models exhausted. Check AWS credentials, region, and model access."
    )


def _parse_stream(event_stream, model_id: str) -> Generator[str, None, None]:
    """
    Parse Bedrock streaming event stream → yield text tokens.
    Handles Claude and Amazon Nova event schemas.
    """
    mid = model_id.lower()
    try:
        for event in event_stream:
            chunk = event.get("chunk")
            if not chunk:
                continue
            data = json.loads(chunk["bytes"].decode("utf-8"))

            # Claude streaming schema
            if "anthropic" in mid:
                if data.get("type") == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield delta.get("text", "")

            # Amazon Nova streaming schema
            elif "nova" in mid or "amazon" in mid:
                output = data.get("contentBlockDelta", {})
                delta = output.get("delta", {})
                if "text" in delta:
                    yield delta["text"]

            # Generic fallback
            else:
                for key in ("text", "generation", "outputText"):
                    if key in data:
                        yield data[key]
                        break
    except Exception as exc:
        log.error("Stream parsing error: %s", exc)
        raise

# ── Context builder ───────────────────────────────────────────────────────────

def _build_rag_context(sources: list[dict]) -> str:
    """
    Build context string for LLM with rich provenance metadata.
    Page numbers and filenames are embedded so the LLM can cite them.
    """
    parts: list[str] = []
    for i, s in enumerate(sources, start=1):
        src      = s.get("source", "unknown")
        drug     = s.get("drug", "")
        reaction = s.get("reaction", "")
        file_    = s.get("file", "")
        page     = s.get("page", "")
        chunk    = s.get("chunk", "")
        score    = s.get("score", 0.0)

        if src == "book":
            # Build citation reference the LLM should use verbatim
            cite_ref = f"Book: {file_}, Page {page}" if page else f"Book: {file_}"
            header = (
                f"[SOURCE {i}] TYPE=BOOK | CITE_AS=[{cite_ref}] | "
                f"File={file_} | Page={page} | Chunk={chunk} | Relevance={score:.3f}"
            )
        else:
            cite_ref = f"FAERS: Drug={drug}, Reaction={reaction}" if drug else "FAERS"
            header = (
                f"[SOURCE {i}] TYPE=FAERS | CITE_AS=[{cite_ref}] | "
                f"Drug={drug} | Reaction={reaction} | Relevance={score:.3f}"
            )

        parts.append(f"--- {header} ---\n{s['text']}\n")

    return "\n".join(parts)


def _build_history_summary_prompt(history: list[Message]) -> str:
    """Build a summarisation prompt for long histories."""
    turns = "\n".join(
        f"{m.role.upper()}: {m.content[:300]}"
        for m in history[:-4]   # summarise everything except last 2 turns
    )
    return (
        f"Summarise this pharmacovigilance conversation concisely "
        f"(max 200 words), preserving key drug names, ADRs, and decisions:\n\n{turns}"
    )


# ── Main Chatbot class ────────────────────────────────────────────────────────

class PVChatbot:
    """
    Production-grade stateful multi-turn Pharmacovigilance chatbot.

    Parameters
    ----------
    top_k          : retrieved chunks per query
    max_history    : max conversation turns kept in context
    aws_region     : override AWS region (default: env AWS_REGION or us-east-1)

    Example
    -------
        bot = PVChatbot()
        r = bot.chat("What are the ADRs of warfarin?")
        print(r.answer)

        r2 = bot.chat("What monitoring is recommended?")   # uses prior context
        print(r2.answer)

        bot.reset()   # start fresh session
    """

    def __init__(
        self,
        top_k: int = MAX_CONTEXT_CHUNKS,
        max_history: int = MAX_HISTORY_TURNS,
        aws_region: Optional[str] = None,
    ) -> None:
        self.top_k       = top_k
        self.max_history = max_history
        self.session_id  = str(uuid.uuid4())
        self.history: list[Message] = []
        self.turn = 0
        self._conversation_summary: Optional[str] = None

        if aws_region:
            os.environ["AWS_REGION"] = aws_region

        log.info("PVChatbot initialised | session=%s", self.session_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, query: str) -> ChatResponse:
        """
        Send a message and get a response.

        Maintains full conversation state across calls.
        """
        t0 = time.perf_counter()
        self.turn += 1
        log.info("[Turn %d] Query: %s", self.turn, query[:100])

        # 1. Classify intent
        intent = classify_intent(query)
        log.info("Intent: %s", intent.value)

        # 2. Retrieve relevant chunks
        sources = retrieve(query, top_k=self.top_k, boost_faers=True)
        context = _build_rag_context(sources)

        # 3. Summarise old history if too long
        if len(self.history) >= SUMMARY_THRESHOLD * 2:
            self._maybe_summarise()

        # 4. Build Bedrock messages payload
        bedrock_messages = self._build_bedrock_messages(query, context)

        # 5. Build dynamic system prompt enriched with intent
        system = self._build_system_prompt(intent)

        # 6. Call Bedrock with retry + fallback
        answer, model_used = _invoke_with_retry(bedrock_messages, system)

        # 7. Update history
        self.history.append(Message(role="user", content=query, intent=intent))
        self.history.append(Message(role="assistant", content=answer))

        # Trim history to sliding window
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

        latency_ms = (time.perf_counter() - t0) * 1000
        log.info(
            "[Turn %d] Answered in %.0fms | model=%s | sources=%d",
            self.turn, latency_ms, model_used, len(sources),
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            intent=intent,
            model_used=model_used,
            session_id=self.session_id,
            turn=self.turn,
            latency_ms=latency_ms,
            retrieved_count=len(sources),
        )


    def stream_chat(self, query: str) -> tuple[Iterator[str], "ChatResponse"]:
        """
        Streaming version of chat().

        Returns (token_iterator, partial_response) where partial_response
        has all fields populated except `answer` (empty string — fill after consuming iterator).

        Usage in Streamlit:
            token_iter, meta = bot.stream_chat(query)
            full_text = st.write_stream(token_iter)
            # history is already updated inside stream_chat

        Returns
        -------
        token_iter  : Iterator[str]  — yields tokens as they arrive
        meta        : ChatResponse   — answer="" until iterator consumed
        """
        t0 = time.perf_counter()
        self.turn += 1
        log.info("[Turn %d | stream] Query: %s", self.turn, query[:100])

        # 1. Classify intent
        intent = classify_intent(query)

        # 2. Retrieve
        sources = retrieve(query, top_k=self.top_k, boost_faers=True)
        context = _build_rag_context(sources)

        # 3. Summarise if needed
        if len(self.history) >= SUMMARY_THRESHOLD * 2:
            self._maybe_summarise()

        # 4. Build payload
        bedrock_messages = self._build_bedrock_messages(query, context)
        system = self._build_system_prompt(intent)

        # 5. Get streaming iterator + model used
        token_iter, model_used = _stream_with_retry(bedrock_messages, system)

        # 6. Wrap iterator to capture full text and update history on completion
        full_text_buffer: list[str] = []

        def _collecting_iter() -> Generator[str, None, None]:
            for token in token_iter:
                full_text_buffer.append(token)
                yield token

            # History updated once stream is fully consumed
            full_answer = "".join(full_text_buffer)
            self.history.append(Message(role="user", content=query, intent=intent))
            self.history.append(Message(role="assistant", content=full_answer))

            if len(self.history) > self.max_history * 2:
                self.history = self.history[-(self.max_history * 2):]

            latency_ms = (time.perf_counter() - t0) * 1000
            log.info("[Turn %d] Stream complete in %.0fms | model=%s",
                     self.turn, latency_ms, model_used)

        meta = ChatResponse(
            answer="",          # filled after stream consumed
            sources=sources,
            intent=intent,
            model_used=model_used,
            session_id=self.session_id,
            turn=self.turn,
            latency_ms=0.0,     # updated in log after stream
            retrieved_count=len(sources),
        )

        return _collecting_iter(), meta

    def reset(self) -> None:
        """Start a new session — clears history and conversation summary."""
        old_session = self.session_id
        self.session_id = str(uuid.uuid4())
        self.history.clear()
        self.turn = 0
        self._conversation_summary = None
        log.info("Session reset | old=%s | new=%s", old_session, self.session_id)

    def get_history(self) -> list[dict]:
        """Return conversation history as plain dicts."""
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
                "intent": m.intent.value if m.intent else None,
            }
            for m in self.history
        ]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _build_system_prompt(self, intent: QueryIntent) -> str:
        """Enrich base system prompt with intent-specific guidance."""
        intent_addendum = {
            QueryIntent.DRUG_SAFETY: (
                "\nFocus: This is a drug safety query. Prioritise FAERS data, "
                "known ADR profiles, and frequency/severity information."
            ),
            QueryIntent.SIGNAL_DETECT: (
                "\nFocus: This is a signal detection query. Discuss statistical "
                "methods, thresholds, and interpretation carefully."
            ),
            QueryIntent.REGULATORY: (
                "\nFocus: This is a regulatory query. Be precise about guideline "
                "names, versions, and jurisdictions (FDA vs EMA vs ICH)."
            ),
            QueryIntent.EPIDEMIOLOGY: (
                "\nFocus: This is a pharmacoepidemiology query. Address study "
                "design, confounding, and limitations of real-world evidence."
            ),
            QueryIntent.GENERAL_PV: (
                "\nFocus: General pharmacovigilance. Use standard PV terminology "
                "and definitions from ICH E2A and GVP modules."
            ),
            QueryIntent.UNKNOWN: "",
        }
        return PV_SYSTEM_PROMPT + intent_addendum.get(intent, "")

    def _build_bedrock_messages(
        self,
        query: str,
        context: str,
    ) -> list[dict]:
        """
        Build the messages list for Bedrock API.

        Structure:
          [optional summary turn]
          [prior conversation turns (sliding window)]
          [current turn with RAG context injected]
        """
        messages: list[dict] = []

        # Inject conversation summary if available
        if self._conversation_summary:
            messages.append({
                "role": "user",
                "content": f"[Conversation summary so far]: {self._conversation_summary}",
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have the context of our previous discussion.",
            })

        # Add recent history (sliding window, skip last 2 — that's the current turn)
        recent = self.history[-(self.max_history * 2):]
        for msg in recent:
            messages.append({"role": msg.role, "content": msg.content})

        # Current query with RAG context
        current_content = (
            f"RELEVANT CONTEXT FROM KNOWLEDGE BASE:\n"
            f"{context}\n"
            f"{'─' * 60}\n"
            f"QUESTION: {query}\n\n"
            f"Please answer based on the context above and our conversation history."
        )
        messages.append({"role": "user", "content": current_content})

        return messages

    def _maybe_summarise(self) -> None:
        """Summarise old conversation turns to preserve token budget."""
        if len(self.history) < SUMMARY_THRESHOLD * 2:
            return

        log.info("Summarising conversation history (%d turns)...", len(self.history))
        summary_prompt = _build_history_summary_prompt(self.history)

        try:
            summary, _ = _invoke_with_retry(
                messages=[{"role": "user", "content": summary_prompt}],
                system="You are a concise summariser. Summarise only, no commentary.",
                model_chain=[BedrockModel.CLAUDE_HAIKU],   # use cheapest for summary
            )
            self._conversation_summary = summary
            # Keep only last 4 turns in history after summarising
            self.history = self.history[-4:]
            log.info("History summarised successfully.")
        except Exception as exc:
            log.warning("Summarisation failed (non-fatal): %s", exc)