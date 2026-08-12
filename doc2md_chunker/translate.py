# translate.py

from __future__ import annotations

import os
import re
import math
import time
import json
from pathlib import Path
from typing import Any, List, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from loguru import logger
from pydantic import BaseModel, Field

from security import get_api_key


ENV_PATH = Path(__file__).resolve().parent.parent / "doc2md_chunker.env"
load_dotenv(ENV_PATH)

HTTPX_TIMEOUT = httpx.Timeout(float(os.getenv("DOC2MD_HTTPX_TIMEOUT", "600.0")))

LLM_MODEL_DEFAULT_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_DEFAULT_BASE_URL", "http://localhost:11434"
).rstrip("/")

LLM_MODEL_CHAT_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_CHAT_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_TRANSLATE_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_TRANSLATE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_CHUNK_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_CHUNK_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_CLASSIFIER_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_CLASSIFIER_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_TABLE_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_TABLE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_OCR_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_OCR_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_PICTURE_BASE_URL = os.getenv(
    "LLM_MODEL_PICTURE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / "doc2md_chunker.env"
load_dotenv(ENV_PATH)


LLM_MODEL_DEFAULT_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_DEFAULT_BASE_URL", "http://ollama:11434"
).rstrip("/")
LLM_MODEL_TRANSLATE_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_TRANSLATE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

TRANSLATE_MODEL = os.getenv("DOC2MD_TRANSLATE_MODEL", os.getenv("DOC2MD_LLM_MODEL", "gemma3:4b"))
OLLAMA_LLM_CONTEXT_LENGTH = int(os.getenv("DOC2MD_LLM_CONTEXT_LENGTH", 4096))
OLLAMA_LLM_NUM_PREDICT = int(os.getenv("DOC2MD_LLM_NUM_PREDICT", 2048))
OLLAMA_LLM_SAFETY_MARGIN_TOKENS = int(os.getenv("DOC2MD_LLM_SAFETY_MARGIN_TOKENS", 512))
OLLAMA_LLM_MAX_RETRIES = int(os.getenv("DOC2MD_LLM_MAX_RETRIES", 2))
OLLAMA_LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("DOC2MD_LLM_RETRY_BACKOFF_SECONDS", 1.5))


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------
# FIX: Declare tags ONLY here on the router.
# Do NOT re-declare tags on individual @router.post() decorators — FastAPI
# merges router-level and route-level tags, which creates duplicate Swagger
# sections (e.g. both "translate" and "Translation" would appear).
# ---------------------------------------------------------------------------
router = APIRouter(
    tags=["Translation"],  # single, canonical tag — Title case for display
    dependencies=[Depends(get_api_key)],
)

# ---------------------------------------------------------------------------
# Token estimation + chunking helpers
# ---------------------------------------------------------------------------

def _contains_cjk(text: str) -> bool:
    return any(_is_cjk(ch) for ch in text)

def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x3040 <= code <= 0x309F
        or 0x30A0 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )

def _estimate_tokens(text: str) -> int:
    """
    Conservative token estimate:
    - CJK: ~1 char/token
    - non-CJK: ~4 chars/token
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    non = len(text) - cjk
    return cjk + math.ceil(non / 4)

def _choose_chars_per_token(text: str) -> float:
    if not text:
        return 4.0
    cjk = sum(1 for ch in text if _is_cjk(ch))
    ratio = cjk / max(1, len(text))
    return 1.2 if ratio >= 0.30 else 4.0

def _compute_max_input_tokens(
    *,
    num_ctx: int,
    num_predict: int,
    system_prompt: str,
    safety_margin_tokens: int,
) -> int:
    overhead = _estimate_tokens(system_prompt) + 64  # chat framing buffer
    budget = num_ctx - num_predict - overhead - safety_margin_tokens
    return max(256, budget)

def _split_by_paragraphs_keep_separators(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(\n{2,})", text)
    return [p for p in parts if p != ""]

def _hard_split_to_fit(piece: str, max_tokens: int) -> List[str]:
    if _estimate_tokens(piece) <= max_tokens:
        return [piece]

    chars_per_token = _choose_chars_per_token(piece)
    target_chars = max(200, int(max_tokens * chars_per_token))

    out: List[str] = []
    start = 0
    n = len(piece)

    while start < n:
        end = min(n, start + target_chars)
        chunk = piece[start:end]

        while end > start + 50 and _estimate_tokens(chunk) > max_tokens:
            end = start + max(50, int((end - start) * 0.85))
            chunk = piece[start:end]

        out.append(chunk)
        start = end

    return out

def _chunk_text_to_token_budget(text: str, max_tokens: int) -> List[str]:
    if not text:
        return []

    pieces = _split_by_paragraphs_keep_separators(text)
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    def flush():
        nonlocal cur, cur_tokens
        if cur:
            chunks.append("".join(cur))
            cur = []
            cur_tokens = 0

    for piece in pieces:
        piece_tokens = _estimate_tokens(piece)

        if piece_tokens > max_tokens:
            flush()
            chunks.extend(_hard_split_to_fit(piece, max_tokens=max_tokens))
            continue

        if cur_tokens + piece_tokens <= max_tokens:
            cur.append(piece)
            cur_tokens += piece_tokens
        else:
            flush()
            cur.append(piece)
            cur_tokens = piece_tokens

    flush()
    return chunks

def _split_markdown_fenced_codeblocks(text: str) -> List[Tuple[str, str]]:
    """
    Split into [("text", ...), ("code", ...), ...].
    Fenced code blocks are passed through unchanged (not translated).
    """
    if not text:
        return []

    lines = text.splitlines(keepends=True)
    segments: List[Tuple[str, str]] = []

    buf: List[str] = []
    mode = "text"
    fence_re = re.compile(r"^\s*```")

    def flush_buf(seg_type: str):
        nonlocal buf
        if buf:
            segments.append((seg_type, "".join(buf)))
            buf = []

    for line in lines:
        if mode == "text":
            if fence_re.match(line):
                flush_buf("text")
                mode = "code"
                buf.append(line)
            else:
                buf.append(line)
        else:
            buf.append(line)
            if fence_re.match(line):
                flush_buf("code")
                mode = "text"

    if buf:
        flush_buf("code" if mode == "code" else "text")

    return segments

# ---------------------------------------------------------------------------
# Ollama call helpers
# ---------------------------------------------------------------------------

def _content_to_ollama_string(content: Any) -> str:
    """
    Ollama /api/chat expects message.content to be a string.
    Some OpenAI-compatible clients send content as a list of parts; normalize here.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)

    return str(content)

def _make_translate_messages(
    system_prompt: str, chunk: str, target_lang: str, target_lang_code: str
) -> list[dict[str, str]]:
    user_prompt = (
        f"Translate the following content into {target_lang} ({target_lang_code}).\n"
        "Return ONLY the translated content.\n"
        "Preserve all Markdown formatting, whitespace, and line breaks.\n"
        "Do NOT add explanations.\n\n"
        "CONTENT START\n"
        f"{chunk}\n"
        "CONTENT END"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

#def _pick_num_predict_for_chunk(chunk: str) -> int:
#    need = int(_estimate_tokens(chunk) * 1.2) + 128
#    return max(256, min(OLLAMA_LLM_NUM_PREDICT, need))

def _pick_num_predict_for_chunk(chunk: str) -> int:
    return -1

def _call_llm_chat(
    messages: list[dict[str, str]],
    model: str,
    num_predict: int,
    base_url: str = LLM_MODEL_TRANSLATE_BASE_URL,
) -> str:
    url = f"{base_url}/v1/chat/completions"
    payload: dict = {
        "model": model,
        "messages": [
            {"role": m["role"], "content": _content_to_ollama_string(m.get("content"))}
            for m in messages
        ],
        "stream": False,
        "temperature": 0.1,
    }
    if num_predict > 0:
        payload["max_tokens"] = num_predict
    with httpx.Client(timeout=HTTPX_TIMEOUT) as client:
        for attempt in range(OLLAMA_LLM_MAX_RETRIES + 1):
            try:
                logger.debug(
                    "[llm] POST {url} | model={model} max_tokens={max_tokens} attempt={attempt}/{max}",
                    url=url,
                    model=model,
                    max_tokens=num_predict,
                    attempt=attempt + 1,
                    max=OLLAMA_LLM_MAX_RETRIES + 1,
                )
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                result = data["choices"][0]["message"]["content"] or ""
                logger.debug(
                    "[llm] Response received | ~{tokens} output tokens",
                    tokens=_estimate_tokens(result),
                )
                return result
            except Exception as exc:
                logger.warning(
                    "[llm] Attempt {attempt}/{max} failed: {exc}",
                    attempt=attempt + 1,
                    max=OLLAMA_LLM_MAX_RETRIES + 1,
                    exc=exc,
                )
                if attempt < OLLAMA_LLM_MAX_RETRIES:
                    sleep_secs = OLLAMA_LLM_RETRY_BACKOFF_SECONDS * (attempt + 1)
                    logger.info("[llm] Retrying in {s:.1f}s ...", s=sleep_secs)
                    time.sleep(sleep_secs)
                    continue
                raise RuntimeError(f"LLM chat failed after {attempt + 1} attempts: {exc}") from exc
    return ""

# ---------------------------------------------------------------------------
# Core translation logic
# ---------------------------------------------------------------------------

def _build_system_prompt(target_lang: str, target_lang_code: str) -> str:
    return (
        f"You are a professional translator. Your task is to translate content into "
        f"{target_lang} ({target_lang_code}) accurately and naturally. "
        "Preserve Markdown formatting, structure, and all technical terms. "
        "Do not add explanations or commentary."
    )

def translate_text(
    text: str,
    target_lang: str = "English",
    target_lang_code: str = "en",
    llm_model: str = TRANSLATE_MODEL,
) -> str:
    system_prompt = _build_system_prompt(target_lang, target_lang_code)
    max_input_tokens = _compute_max_input_tokens(
        num_ctx=OLLAMA_LLM_CONTEXT_LENGTH,
        num_predict=OLLAMA_LLM_NUM_PREDICT,
        system_prompt=system_prompt,
        safety_margin_tokens=OLLAMA_LLM_SAFETY_MARGIN_TOKENS,
    )

    total_estimated_tokens = _estimate_tokens(text)
    logger.info(
        "[translate_text] Starting | lang={lang} ({code}) | model={model} | "
        "text_chars={chars} ~{tokens} tokens | context={ctx} num_predict={np} "
        "safety_margin={sm} => max_input_tokens={mit}",
        lang=target_lang,
        code=target_lang_code,
        model=llm_model,
        chars=len(text),
        tokens=total_estimated_tokens,
        ctx=OLLAMA_LLM_CONTEXT_LENGTH,
        np=OLLAMA_LLM_NUM_PREDICT,
        sm=OLLAMA_LLM_SAFETY_MARGIN_TOKENS,
        mit=max_input_tokens,
    )

    chunks = _chunk_text_to_token_budget(text, max_input_tokens)
    total_chunks = len(chunks)
    logger.info(
        "[translate_text] Split into {n} chunk(s) (max_input_tokens={mit})",
        n=total_chunks,
        mit=max_input_tokens,
    )

    translated_parts: List[str] = []
    t0_total = time.monotonic()

    for idx, chunk in enumerate(chunks, start=1):
        chunk_tokens = _estimate_tokens(chunk)
        num_predict = _pick_num_predict_for_chunk(chunk)
        logger.info(
            "[translate_text] Chunk {idx}/{total} | ~{tokens} input tokens | num_predict={np}",
            idx=idx,
            total=total_chunks,
            tokens=chunk_tokens,
            np=num_predict,
        )
        t0 = time.monotonic()
        messages = _make_translate_messages(system_prompt, chunk, target_lang, target_lang_code)
        translated_chunk = _call_llm_chat(messages, llm_model, num_predict)
        elapsed = time.monotonic() - t0
        logger.info(
            "[translate_text] Chunk {idx}/{total} done in {elapsed:.1f}s | "
            "output ~{out_tokens} tokens",
            idx=idx,
            total=total_chunks,
            elapsed=elapsed,
            out_tokens=_estimate_tokens(translated_chunk),
        )
        translated_parts.append(translated_chunk)

    total_elapsed = time.monotonic() - t0_total
    logger.info(
        "[translate_text] Complete | {n} chunk(s) translated in {elapsed:.1f}s total",
        n=total_chunks,
        elapsed=total_elapsed,
    )
    return "".join(translated_parts)

def translate_text_file(
    file_path: Path,
    target_lang: str = "English",
    target_lang_code: str = "en",
    llm_model: str = TRANSLATE_MODEL,
) -> str:
    logger.info("[translate_text_file] Reading file: {path}", path=file_path)
    text = file_path.read_text(encoding="utf-8")
    logger.info("[translate_text_file] File loaded | {chars} chars", chars=len(text))
    return translate_text(text, target_lang, target_lang_code, llm_model)

def translate_markdown(
    text: str,
    target_lang: str = "English",
    target_lang_code: str = "en",
    keep_media_links: bool = True,
    llm_model: str = TRANSLATE_MODEL,
) -> str:
    """Translate a Markdown string, preserving fenced code blocks unchanged."""
    segments = _split_markdown_fenced_codeblocks(text)
    text_segments = sum(1 for t, _ in segments if t == "text")
    code_segments = sum(1 for t, _ in segments if t == "code")
    logger.info(
        "[translate_markdown] {total} segment(s) found: {ts} text, {cs} fenced code block(s) "
        "(code blocks are NOT translated)",
        total=len(segments),
        ts=text_segments,
        cs=code_segments,
    )

    result_parts: List[str] = []
    for seg_idx, (seg_type, seg_text) in enumerate(segments, start=1):
        if seg_type == "code":
            logger.info(
                "[translate_markdown] Segment {idx}/{total}: code block ({chars} chars) — "
                "passing through unchanged",
                idx=seg_idx,
                total=len(segments),
                chars=len(seg_text),
            )
            result_parts.append(seg_text)
        else:
            logger.info(
                "[translate_markdown] Segment {idx}/{total}: text ({chars} chars "
                "~{tokens} tokens) — translating",
                idx=seg_idx,
                total=len(segments),
                chars=len(seg_text),
                tokens=_estimate_tokens(seg_text),
            )
            result_parts.append(
                translate_text(seg_text, target_lang, target_lang_code, llm_model)
            )

    logger.info("[translate_markdown] All segments processed.")
    return "".join(result_parts)

def translate_markdown_file(
    file_path: Path,
    target_lang: str = "English",
    target_lang_code: str = "en",
    keep_media_links: bool = True,
    llm_model: str = TRANSLATE_MODEL,
) -> str:
    logger.info(
        "[translate_markdown_file] Reading file: {path} | target_lang={lang} ({code}) | model={model}",
        path=file_path,
        lang=target_lang,
        code=target_lang_code,
        model=llm_model,
    )
    text = file_path.read_text(encoding="utf-8")
    logger.info(
        "[translate_markdown_file] File loaded | {chars} chars (~{tokens} estimated tokens) | "
        "context_length={ctx}",
        chars=len(text),
        tokens=_estimate_tokens(text),
        ctx=OLLAMA_LLM_CONTEXT_LENGTH,
    )
    return translate_markdown(text, target_lang, target_lang_code, keep_media_links, llm_model)

def detect_language(text: str, llm_model: str = TRANSLATE_MODEL) -> tuple[str, float]:
    sample = text[:2000]
    prompt = (
        "Detect the language of the following text. "
        'Reply with a JSON object: {"language": "", "confidence": <0.0-1.0>}\n\n'
        f"TEXT:\n{sample}"
    )

    messages = [
        {"role": "system", "content": "You are a language detection expert. Reply with valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    logger.debug(
        "[detect_language] Detecting language of {chars}-char sample with model={model}",
        chars=len(sample),
        model=llm_model,
    )
    raw = _call_llm_chat(messages, llm_model, num_predict=64)
    try:
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            lang = str(data.get("language", "unknown"))
            conf = float(data.get("confidence", 0.0))
            logger.info(
                "[detect_language] Detected: {lang} (confidence={conf:.2f})",
                lang=lang,
                conf=conf,
            )
            return lang, conf
    except Exception:
        pass
    logger.warning("[detect_language] Failed to parse language detection response: {raw}", raw=raw)
    return "unknown", 0.0

def detect_language_file(
    file_path: Path, llm_model: str = TRANSLATE_MODEL
) -> tuple[str, float]:
    logger.info("[detect_language_file] Reading file: {path}", path=file_path)
    text = file_path.read_text(encoding="utf-8")
    return detect_language(text, llm_model)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TranslateMarkdownRequest(BaseModel):
    markdown: str = Field(..., description="Markdown content to translate")
    target_lang: str = Field("English", description="Target language.")
    target_lang_code: str = Field("en", description="Target language code (BCP-47 / ISO, e.g., 'en', 'fr', 'es')")
    keep_media_links: bool = Field(True, description="Whether to keep Markdown image links in the translated output")
    llm_model: str = Field(TRANSLATE_MODEL, description="LLM model name to use for translation")

class TranslateTextRequest(BaseModel):
    markdown: str = Field(..., description="Text content to translate (plain text or Markdown)")
    target_lang: str = Field("English", description="Target language.")
    target_lang_code: str = Field("en", description="Target language code (BCP-47 / ISO, e.g., 'en', 'fr', 'es')")
    keep_media_links: bool = Field(True, description="Whether to keep Markdown image links in the translated output")
    llm_model: str = Field(TRANSLATE_MODEL, description="LLM model name to use for translation")

class TranslateTextResponse(BaseModel):
    translated_text: str = Field(..., description="The translated text in the target language")

class DetectLanguageRequest(BaseModel):
    text: str = Field(..., description="Text content to analyze for language detection")
    llm_model: str = Field(TRANSLATE_MODEL, description="LLM model name to use for language detection")

class DetectLanguageResponse(BaseModel):
    language: str = Field(..., description="Detected language code (BCP-47 / ISO, e.g., 'en', 'fr', 'es')")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)

class DetectLanguageFileResponse(BaseModel):
    language: str = Field(..., description="Detected language code (BCP-47 / ISO)")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0", ge=0.0, le=1.0)
    filename: str = Field(..., description="Name of the analyzed file")

class TranslateFileResponse(BaseModel):
    translated_text: str = Field(..., description="The translated file content")
    filename: str = Field(..., description="Name of the translated file")

# ---------------------------------------------------------------------------
# API endpoints
# NOTE: tags are intentionally NOT set here — they inherit from the router
# (tags=["Translation"]). Adding tags here would cause FastAPI to merge
# them, resulting in duplicate Swagger UI sections.
# ---------------------------------------------------------------------------

@router.post(
    "/translate-text",
    response_model=TranslateTextResponse,
    summary="Translate raw text",
    response_description="Translated text in the target language",
)
async def api_translate_text(req: TranslateTextRequest):
    """Translate a block of plain or Markdown text into the specified target language."""
    logger.info(
        "[api_translate_text] Request | lang={lang} ({code}) | model={model} | chars={chars}",
        lang=req.target_lang,
        code=req.target_lang_code,
        model=req.llm_model,
        chars=len(req.markdown),
    )
    try:
        translated = await run_in_threadpool(
            translate_text,
            req.markdown,
            req.target_lang,
            req.target_lang_code,
            req.llm_model,
        )
    except Exception as e:
        logger.exception(f"translate_text failed: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")
    logger.info("[api_translate_text] Done | output chars={chars}", chars=len(translated))
    return TranslateTextResponse(translated_text=translated)

@router.post(
    "/translate-markdown",
    response_model=TranslateTextResponse,
    summary="Translate a Markdown string",
    response_description="Translated Markdown content with fenced code blocks preserved",
)
async def api_translate_markdown(req: TranslateMarkdownRequest):
    """
    Translate a Markdown string passed directly in the request body.
    Fenced code blocks (``` ... ```) are preserved as-is and never translated.
    """
    logger.info(
        "[api_translate_markdown] Request | lang={lang} ({code}) | model={model} | chars={chars}",
        lang=req.target_lang,
        code=req.target_lang_code,
        model=req.llm_model,
        chars=len(req.markdown),
    )
    try:
        translated = await run_in_threadpool(
            translate_markdown,
            req.markdown,
            req.target_lang,
            req.target_lang_code,
            req.keep_media_links,
            req.llm_model,
        )
    except Exception as e:
        logger.exception(f"translate_markdown failed: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")
    logger.info("[api_translate_markdown] Done | output chars={chars}", chars=len(translated))
    return TranslateTextResponse(translated_text=translated)

@router.post(
    "/detect-language",
    response_model=DetectLanguageResponse,
    summary="Detect language of raw text",
    response_description="Detected language code and confidence score",
)
async def api_detect_language(req: DetectLanguageRequest):
    """Detect the BCP-47 language code of a text string and return a confidence score."""
    try:
        lang, conf = await run_in_threadpool(detect_language, req.text, req.llm_model)
    except Exception as e:
        logger.exception(f"detect_language failed: {e}")
        raise HTTPException(status_code=500, detail="Language detection failed")
    return DetectLanguageResponse(language=lang, confidence=conf)

@router.post(
    "/translate-text-file",
    response_model=TranslateFileResponse,
    summary="Translate a plain text file",
    response_description="Translated file content and original filename",
)
async def api_translate_text_file(
    file: UploadFile = File(..., description="Plain text file to translate."),
    target_lang: str = Query("English", description="Target language."),
    target_lang_code: str = Query("en", description="Target language (BCP-47 / ISO code)."),
    llm_model: str = Query(TRANSLATE_MODEL, description="LLM model name for translation."),
):
    """Upload a plain `.txt` file and receive the full translated content."""
    logger.info(
        "[api_translate_text_file] Received file={filename} | lang={lang} ({code}) | model={model}",
        filename=file.filename,
        lang=target_lang,
        code=target_lang_code,
        model=llm_model,
    )
    try:
        data = await file.read()
        tmp_path = Path(f"/tmp/{file.filename}")
        tmp_path.write_text(data.decode("utf-8", errors="replace"), encoding="utf-8")
        logger.info(
            "[api_translate_text_file] File written to {path} | {bytes} bytes",
            path=tmp_path,
            bytes=len(data),
        )
    except Exception as e:
        logger.exception(f"Failed to read uploaded file for translate_text_file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    try:
        translated = await run_in_threadpool(
            translate_text_file,
            tmp_path,
            target_lang,
            target_lang_code,
            llm_model,
        )
    except Exception as e:
        logger.exception(f"translate_text_file failed: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")

    logger.info("[api_translate_text_file] Done | output chars={chars}", chars=len(translated))
    return TranslateFileResponse(translated_text=translated, filename=file.filename)

@router.post(
    "/translate-markdown-file",
    response_model=TranslateFileResponse,
    summary="Translate a Markdown file",
    response_description="Translated Markdown content and original filename",
)
async def api_translate_markdown_file(
    file: UploadFile = File(..., description="Markdown (.md) file to translate."),
    target_lang_code: str = Query("en", description="Target language (BCP-47 / ISO code)."),
    target_lang: str = Query("English", description="Target language."),
    keep_media_links: bool = Query(True, description="Whether to keep markdown image links in the translated output."),
    llm_model: str = Query(TRANSLATE_MODEL, description="LLM model name for translation."),
):
    """
    Upload a Markdown `.md` file and receive a fully translated version.
    Fenced code blocks (``` ... ```) are preserved as-is and never translated.
    The file is automatically split into chunks that fit within OLLAMA_LLM_CONTEXT_LENGTH
    tokens, so large documents are translated across multiple LLM calls transparently.
    """
    logger.info(
        "[api_translate_markdown_file] Received file={filename} | lang={lang} ({code}) | "
        "model={model} | context_length={ctx}",
        filename=file.filename,
        lang=target_lang,
        code=target_lang_code,
        model=llm_model,
        ctx=OLLAMA_LLM_CONTEXT_LENGTH,
    )
    try:
        data = await file.read()
        tmp_path = Path(f"/tmp/{file.filename}")
        tmp_path.write_text(data.decode("utf-8", errors="replace"), encoding="utf-8")
        logger.info(
            "[api_translate_markdown_file] File written to {path} | {bytes} bytes "
            "(~{tokens} estimated tokens)",
            path=tmp_path,
            bytes=len(data),
            tokens=_estimate_tokens(data.decode("utf-8", errors="replace")),
        )
    except Exception as e:
        logger.exception(f"Failed to read uploaded file for translate_markdown_file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    try:
        translated = await run_in_threadpool(
            translate_markdown_file,
            tmp_path,
            target_lang,
            target_lang_code,
            keep_media_links,
            llm_model,
        )
    except Exception as e:
        logger.exception(f"translate_markdown_file failed: {e}")
        raise HTTPException(status_code=500, detail="Translation failed")

    logger.info(
        "[api_translate_markdown_file] Done | output chars={chars}",
        chars=len(translated),
    )
    return TranslateFileResponse(translated_text=translated, filename=file.filename)

@router.post(
    "/detect-language-file",
    response_model=DetectLanguageFileResponse,
    summary="Detect language of a text/Markdown file",
    response_description="Detected language, confidence score, and filename",
)
async def api_detect_language_file(
    file: UploadFile = File(..., description="Text/markdown file for language detection."),
    llm_model: str = Query(TRANSLATE_MODEL, description="LLM model name for language detection."),
):
    """Upload a text or Markdown file and detect the language used within it."""
    try:
        data = await file.read()
        tmp_path = Path(f"/tmp/{file.filename}")
        tmp_path.write_text(data.decode("utf-8", errors="replace"), encoding="utf-8")
    except Exception as e:
        logger.exception(f"Failed to read uploaded file for detect_language_file: {e}")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")

    try:
        lang, conf = await run_in_threadpool(detect_language_file, tmp_path, llm_model)
    except Exception as e:
        logger.exception(f"detect_language_file failed: {e}")
        raise HTTPException(status_code=500, detail="Language detection failed")

    return DetectLanguageFileResponse(language=lang, confidence=conf, filename=file.filename)
