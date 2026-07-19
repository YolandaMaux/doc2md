# utils.py
from __future__ import annotations

import asyncio
import os
import re
import base64
import tempfile
import threading
import json
from io import BytesIO
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from loguru import logger
from pydantic import BaseModel, Field

ENV_PATH = Path(__file__).resolve().parent.parent / "doc2md_chunker.env"
load_dotenv(ENV_PATH)

HTTPX_TIMEOUT = httpx.Timeout(float(os.getenv("HTTPX_TIMEOUT", "600.0")))

LLM_MODEL_DEFAULT_BASE_URL = os.getenv(
    "LLM_MODEL_DEFAULT_BASE_URL", "http://localhost:8080"
).rstrip("/")

LLM_MODEL_CHAT_BASE_URL = os.getenv(
    "LLM_MODEL_CHAT_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_TRANSLATE_BASE_URL = os.getenv(
    "LLM_MODEL_TRANSLATE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_CHUNK_BASE_URL = os.getenv(
    "LLM_MODEL_CHUNK_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_CLASSIFIER_BASE_URL = os.getenv(
    "LLM_MODEL_CLASSIFIER_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_TABLE_BASE_URL = os.getenv(
    "LLM_MODEL_TABLE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_OCR_BASE_URL = os.getenv(
    "LLM_MODEL_OCR_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_PICTURE_BASE_URL = os.getenv(
    "LLM_MODEL_PICTURE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")

LLM_MODEL_FIGURE_BASE_URL = os.getenv(
    "LLM_MODEL_FIGURE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL
).rstrip("/")
# ---------------------------------------------------------------------------
# Environment / defaults
# ---------------------------------------------------------------------------
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_OCR_TEMPERATURE", "0.1"))

IMAGE_EXTS = set(
    os.getenv("IMAGE_EXTS", ".png,.jpg,.jpeg,.webp,.bmp,.tiff").split(",")
)

MODEL_CLASSIFIER = os.getenv("LLM_MODEL_CLASSIFIER", "llama3.2-vision:11b")
MODEL_OCR = os.getenv("LLM_MODEL_OCR", "glm-ocr:bf16")
MODEL_TABLE = os.getenv("LLM_MODEL_TABLE", "glm-ocr:bf16")
MODEL_FIGURE = os.getenv("LLM_MODEL_FIGURE", "glm-ocr:bf16")
MODEL_PICTURE = os.getenv("LLM_MODEL_PICTURE", "llama3.2-vision:11b")

# ---------------------------------------------------------------------------
# /media_to_text — Whisper defaults (override via .env)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
AUDIO_EXTS = set(
    os.getenv("AUDIO_EXTS", ".mp3,.wav,.flac,.ogg,.m4a,.aac,.wma,.opus,.webm").split(",")
)

VIDEO_EXTS = set(
    os.getenv("VIDEO_EXTS", ".mp4,.mkv,.avi,.mov,.flv,.wmv,.ts,.3gp").split(",")
)

MEDIA_EXTS = AUDIO_EXTS | VIDEO_EXTS

_sync_http_client = httpx.Client(timeout=HTTPX_TIMEOUT)
router = APIRouter()

# ---------------------------------------------------------------------------
# Pydantic v2 sentinel guard
# ---------------------------------------------------------------------------
# Pydantic v2 uses PydanticUndefinedType as the sentinel for "no default set".
# This is NOT the same as Python's Ellipsis (...), so we must guard for both
# when introspecting FieldInfo.default — otherwise JSON serialization crashes
# with: PydanticSerializationError: Unable to serialize unknown type:
#         <class 'pydantic_core._pydantic_core.PydanticUndefinedType'>
try:
    from pydantic_core import PydanticUndefinedType as _PydanticUndefinedType
    _UNDEFINED_TYPES = (_PydanticUndefinedType,)
except ImportError:
    _UNDEFINED_TYPES = ()


def _is_undefined(value: Any) -> bool:
    """Return True if *value* is any known 'no default' sentinel."""
    if value is ...:
        return True
    if _UNDEFINED_TYPES and isinstance(value, _UNDEFINED_TYPES):
        return True
    return False


def _safe_default(value: Any) -> Any:
    """Coerce undefined sentinels to None so they are JSON-serialisable."""
    return None if _is_undefined(value) else value


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _iso_to_ymd(date_str: str | None) -> str | None:
    if not date_str:
        return None
    s = date_str.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y/%m/%d")
    except Exception:
        return None


def _bytes_to_gb(size_bytes: int | float | None) -> float | None:
    if size_bytes is None:
        return None
    try:
        return round(float(size_bytes) / (1024 ** 3), 3)
    except Exception:
        return None


def _slugify(text: str) -> str:
    """Convert a title string to a URL/anchor-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


async def enforce_max_upload_size(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> None:
    length_header = file.headers.get("content-length")
    if not length_header:
        return
    try:
        length = int(length_header)
    except ValueError:
        return
    if length > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({length} bytes). Max allowed is {max_bytes} bytes.",
        )


# ---------------------------------------------------------------------------
# /features
# ---------------------------------------------------------------------------
@router.get(
    "/features",
    summary="List available API features",
    description=(
        "Returns a comprehensive list of all available API endpoints with their configuration. "
        "Introspects the FastAPI application and returns routes, methods, and parameters."
    ),
    tags=["Utility"],
    response_description="List of all API features with their configurations",
)
async def list_features(request: Request) -> Dict[str, Any]:
    features: List[Dict[str, Any]] = []
    for route in request.app.routes:
        if not isinstance(route, APIRoute):
            continue
        methods = sorted(m for m in (route.methods or []) if m not in {"HEAD", "OPTIONS"})
        if not methods:
            continue
        feature: Dict[str, Any] = {
            "path": route.path,
            "methods": methods,
            "name": route.name,
            "summary": (route.summary or "").strip() or None,
            "description": (route.description or "").strip() or None,
            "endpoint": f"{route.endpoint.__module__}.{route.endpoint.__name__}",
            "parameters": [],
        }

        for dep in route.dependant.query_params + route.dependant.body_params:
            # ------------------------------------------------------------------
            # FastAPI 0.100+ (Pydantic v2): ModelField dropped .required /
            # .type_ / .default — use safe getattr with sensible fallbacks.
            # ------------------------------------------------------------------

            # ── required ──────────────────────────────────────────────────────
            _required = getattr(dep, "required", None)
            if _required is None:
                _fi = getattr(dep, "field_info", None)
                if _fi is not None:
                    import inspect as _inspect
                    _req_fn = getattr(_fi, "is_required", None)
                    if callable(_req_fn):
                        _required = _req_fn()
                    else:
                        # field has no default → required
                        _required = _is_undefined(getattr(_fi, "default", ...))
                else:
                    _required = True  # conservative fallback

            # ── type name ─────────────────────────────────────────────────────
            _type = getattr(dep, "type_", None)
            if _type is None:
                _type = getattr(dep, "annotation", None)
            _type_name = getattr(_type, "__name__", str(_type)) if _type is not None else "unknown"

            # ── default ───────────────────────────────────────────────────────
            # Guard against BOTH Ellipsis (...) AND Pydantic v2's
            # PydanticUndefinedType — neither is JSON-serialisable.
            _default = _safe_default(getattr(dep, "default", ...))
            if _default is None:
                _fi = getattr(dep, "field_info", None)
                _raw = getattr(_fi, "default", ...) if _fi is not None else ...
                _default = _safe_default(_raw)

            feature["parameters"].append({
                "name": dep.name,
                "required": _required,
                "type": _type_name,
                "default": _default,
            })
        features.append(feature)
    return {"features": features}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    summary="Health check endpoint",
    description="Simple health check to verify the API service is running and responsive.",
    tags=["Utility"],
    response_description="Service health status",
)
async def health() -> Dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# LLM model listing
# ---------------------------------------------------------------------------
class LLMModelInfo(BaseModel):
    name: str = Field(..., description="Model tag name, e.g. 'llama3.2:latest'.")
    model: Optional[str] = Field(None, description="Model identifier.")
    modified: Optional[str] = Field(None, description="Date formatted as YYYY/MM/DD.")
    size_GB: Optional[float] = Field(None, description="Model size in gigabytes.")
    parameters: Optional[str] = Field(None, description="Parameter size string, e.g. '7B'.")


# def list_llm_models(base_url: str = LLM_MODEL_DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
#     url = f"{base_url.rstrip('/')}/v1/models"
#     resp = _sync_http_client.get(url)
#     resp.raise_for_status()
#     data = resp.json() or {}
#     out = []
#     for m in data.get("data", []):   # OpenAI shape: {"object":"list","data":[...]}
#         out.append({
#             "name": m.get("id"),
#             "model": m.get("id"),
#             "modified": _iso_to_ymd(str(m.get("created", ""))),
#             "size_GB": None,
#             "parameters": None,
#         })
#     return out


# def running_llm_models(base_url: str = LLM_MODEL_DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
#     """List running (loaded) LLM models via GET /api/ps."""
#     url = f"{base_url.rstrip('/')}/api/ps"
#     resp = _sync_http_client.get(url)
#     resp.raise_for_status()
#     data = resp.json() or {}
#     out: List[Dict[str, Any]] = []
#     for m in data.get("models", []) or []:
#         details = m.get("details") or {}
#         out.append({
#             "name": m.get("name"),
#             "model": m.get("model"),
#             "modified": _iso_to_ymd(m.get("expires_at")),
#             "size_GB": _bytes_to_gb(m.get("size")),
#             "parameters": details.get("parameter_size"),
#         })
#     return out


def _normalize_openai_model_entry(m: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": m.get("id"),
        "model": m.get("id"),
        "modified": _iso_to_ymd(str(m.get("created", ""))),
        "size_GB": None,
        "parameters": None,
    }


def list_llm_models(base_url: str = LLM_MODEL_DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v1/models"
    resp = _sync_http_client.get(url)
    resp.raise_for_status()
    data = resp.json() or {}
    return [_normalize_openai_model_entry(m) for m in data.get("data", [])]


def running_llm_models(base_url: str = LLM_MODEL_DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
    """
    Return currently running/loaded models when the backend exposes that concept.

    Backend behavior:
    - Ollama: uses GET /api/ps
    - OpenAI-compatible / llama.cpp servers: falls back to GET /v1/models
      because many such servers do not expose a separate 'running models' API
    """
    base = base_url.rstrip("/")

    # 1) Ollama loaded models endpoint
    try:
        resp = _sync_http_client.get(f"{base}/api/ps")
        if resp.status_code < 400:
            data = resp.json() or {}
            out: List[Dict[str, Any]] = []
            for m in data.get("models", []) or []:
                details = m.get("details") or {}
                out.append({
                    "name": m.get("name"),
                    "model": m.get("model"),
                    "modified": _iso_to_ymd(m.get("expires_at")),
                    "size_GB": _bytes_to_gb(m.get("size")),
                    "parameters": details.get("parameter_size"),
                })
            return out
    except httpx.HTTPError:
        pass

    # 2) OpenAI-compatible fallback, including llama.cpp server
    try:
        resp = _sync_http_client.get(f"{base}/v1/models")
        resp.raise_for_status()
        data = resp.json() or {}
        return [_normalize_openai_model_entry(m) for m in data.get("data", [])]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to determine running/available models from backend. "
                "Tried Ollama /api/ps and OpenAI-compatible /v1/models. "
                f"Last error: {exc}"
            ),
        ) from exc

@router.get(
    "/list_llm_models",
    summary="List installed LLM models",
    description=(
        "Calls LLM GET /api/tags and returns a simplified model list. "
        "Fields: name, model, modified (YYYY/MM/DD), size_GB, parameters."
    ),
    tags=["Utility"],
    response_model=List[LLMModelInfo],
)
async def list_llm_models_endpoint(
    base_url: str = Query(LLM_MODEL_DEFAULT_BASE_URL, description="LLM base URL (default: LLM_MODEL_DEFAULT_BASE_URL)."),
):
    return list_llm_models(base_url=base_url)


@router.get(
    "/running_llm_models",
    summary="List running or available LLM models",
    description=(
        "Returns models from the configured backend. "
        "Uses Ollama `/api/ps` when available; otherwise falls back to "
        "OpenAI-compatible `/v1/models` for backends such as llama.cpp. "
        "Fields: name, model, modified (YYYY/MM/DD), size_GB, parameters."
    ),
    tags=["Utility"],
    response_model=List[LLMModelInfo],
)
async def running_llm_models_endpoint(
    base_url: str = Query(
        LLM_MODEL_DEFAULT_BASE_URL,
        description="LLM base URL (Ollama or OpenAI-compatible, including llama.cpp).",
    ),
):
    return running_llm_models(base_url=base_url)


# ---------------------------------------------------------------------------
# /image-description — vision LLM helpers
# ---------------------------------------------------------------------------
class ImageCategory(str, Enum):
    """Supported image content types for classification and description."""
    picture = "Picture"
    text = "Text"
    table = "Table"
    figure = "Figure"


def _encode_image_b64(image_path: Path) -> str:
    with image_path.open("rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def _call_vision_llm(
    *,
    prompt: str,
    image_b64: str,
    llm_model: str,
    temperature: float,
    base_url: str,
    system_prompt: str = "You are a helpful assistant that analyses images with precision and accuracy.",
) -> str:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            },
        ],
        "stream": False,
        "temperature": temperature,
    }
    logger.info(f"Calling vision LLM url={url} model={llm_model}")
    try:
        resp = _sync_http_client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.exception(f"Vision LLM call failed: {exc}")
        raise HTTPException(status_code=500, detail=f"LLM call failed: {exc}") from exc


def classify_image(image_b64: str, classifier_model: str, temperature: float) -> ImageCategory:
    prompt = (
        "Classify this image into exactly one of the following categories:\n"
        "- Picture : a photograph, illustration, artwork, or visual scene\n"
        "- Text : an image whose primary content is readable text (e.g. a scanned page, screenshot)\n"
        "- Table : a structured grid / data table\n"
        "- Figure : a chart, graph, diagram, plot, or schematic\n\n"
        "Reply with ONLY the single category word (Picture, Text, Table, or Figure). No explanation."
    )
    system = "You are a strict image classifier. Respond with exactly one word: Picture, Text, Table, or Figure."
    raw = _call_vision_llm(
        prompt=prompt,
        image_b64=image_b64,
        llm_model=classifier_model,
        temperature=temperature,
        base_url=LLM_MODEL_CLASSIFIER_BASE_URL,
        system_prompt=system,
    ).strip()
    for cat in ImageCategory:
        if cat.value.lower() == raw.lower():
            return cat
    for cat in ImageCategory:
        if cat.value.lower() in raw.lower():
            return cat
    logger.warning(f"Unrecognised category {raw!r} – defaulting to Picture.")
    return ImageCategory.picture


# ---------------------------------------------------------------------------
# Per-category description functions
# Output: LLM-chunking-friendly Markdown with HTML comment anchors
# ---------------------------------------------------------------------------
def _describe_picture(image_b64: str, model: str, temperature: float, user_prompt: Optional[str] = None) -> str:
    """
    **Picture:**

    <!-- picture-{slug} -->
    Picture: {title}
    {~100 word description}
    <!-- end picture-{slug} -->
    """
    prompt = (
        "Describe this image in detail in approximately 100 words. "
        "First line must be a short descriptive title prefixed with TITLE: "
        "(e.g. TITLE: office-team-meeting). "
        "Then write the description as plain prose. "
        "Focus only on what is clearly visible. "
        "No preamble, no labels beyond the TITLE line."
    )
    if user_prompt:
        prompt += f"\n\n{user_prompt}"


    raw = _call_vision_llm(prompt=prompt, image_b64=image_b64, llm_model=model, temperature=temperature, base_url=LLM_MODEL_PICTURE_BASE_URL).strip()
    title, description = "picture", raw
    for i, line in enumerate(raw.splitlines()):
        if line.upper().startswith("TITLE"):
            title = line.split(":", 1)[1].strip() or title
            description = "\n".join(raw.splitlines()[i + 1:]).strip()
            break
    slug = _slugify(title)
    return (
        f"<!-- picture-{slug} -->\n"
        f"Picture: {title}\n"
        f"{description}\n"
        f"<!-- end picture-{slug} -->"
    )


def _describe_table(image_b64: str, model: str, temperature: float) -> str:
    """
    **Table:**

    <!-- table-{slug} -->
    Table: {title}
    | col | col |
    | --- | --- |
    <!-- end table-{slug} -->
    """
    prompt = (
        "This image contains a table. Do the following:\n"
        "1. Write a short, descriptive title on the first line prefixed with TITLE: "
        "(e.g. TITLE: Monthly Sales by Region).\n"
        "2. Reproduce the full table in valid GitHub-Flavoured Markdown (GFM):\n"
        "   - Use real column names from the table as the header row.\n"
        "   - Remove empty trailing columns.\n"
        "   - Align numeric columns with ---: and text columns with :---.\n"
        "Output ONLY the TITLE line followed immediately by the GFM table. No extra commentary."
    )
    raw = _call_vision_llm(prompt=prompt, image_b64=image_b64, llm_model=model, temperature=temperature, base_url=LLM_MODEL_TABLE_BASE_URL).strip()
    title, tablebody = "Table", raw
    for i, line in enumerate(raw.splitlines()):
        if line.upper().startswith("TITLE"):
            title = line.split(":", 1)[1].strip() or title
            tablebody = "\n".join(raw.splitlines()[i + 1:]).strip()
            break
    slug = _slugify(title)
    return (
        f"<!-- table-{slug} -->\n"
        f"Table: {title}\n"
        f"{tablebody}\n"
        f"<!-- end table-{slug} -->"
    )


def _describe_figure(image_b64: str, model: str, temperature: float) -> str:
    prompt = (
        "This image contains a chart, graph, diagram, or figure. Do the following:\n"
        "1. Write a short descriptive title on the first line prefixed with TITLE: "
        "(e.g. TITLE: Q3 Revenue by Product Line).\n"
        "2. On the next line write the chart/diagram type prefixed with TYPE: "
        "(e.g. TYPE: Bar Chart).\n"
        "3. Write a concise description (under 100 words) prefixed with DESCRIPTION: on its own line, "
        "covering key trends, axes, legend items, and notable values.\n"
        "No commentary outside these three sections."
    )
    raw = _call_vision_llm(
        prompt=prompt,
        image_b64=image_b64,
        llm_model=model,
        temperature=temperature,
        base_url=LLM_MODEL_FIGURE_BASE_URL,
    ).strip()

    lines = raw.splitlines()
    title = "Figure"
    charttype = None
    description = raw

    for line in lines:
        if line.upper().startswith("TITLE"):
            title = line.split(":", 1)[1].strip() or title
            break

    for line in lines:
        if line.upper().startswith("TYPE"):
            charttype = line.split(":", 1)[1].strip() or None
            break

    for i, line in enumerate(lines):
        if line.upper().startswith("DESCRIPTION"):
            first = line.split(":", 1)[1].strip()
            rest = "\n".join(lines[i + 1:]).strip()
            description = (first + "\n" + rest).strip() if rest else first.strip()
            break

    typeline = f"Type: {charttype}\n" if charttype else ""
    return f"<!-- figure-{_slugify(title)} -->\nFigure: {title}\n{typeline}{description}\n<!-- end figure-{_slugify(title)} -->"


def _describe_text_ocr(image_b64: str, model: str, temperature: float) -> str:
    """
    **OCR / Text:**

    <!-- ocr-{slug} -->
    OCR Text: {label}
    {verbatim text}
    <!-- end ocr-{slug} -->
    """
    prompt = (
        "Extract ALL text from this image exactly as it appears. "
        "First line must be a short source label prefixed with LABEL: "
        "(e.g. LABEL: page-1 or LABEL: invoice-header). "
        "Then reproduce the full text preserving line breaks, indentation, columns, "
        "bullet points, numbering, and any other visible formatting. "
        "If no readable text exists, reply with LABEL: no-text-found then NO TEXT FOUND. "
        "Output ONLY the LABEL line followed by the extracted text."
    )
    system = "You are a high-accuracy OCR engine. Reproduce text verbatim, maintaining original structure."
    raw = _call_vision_llm(
        prompt=prompt, image_b64=image_b64, llm_model=model,
        temperature=temperature, base_url=LLM_MODEL_OCR_BASE_URL, system_prompt=system,
    ).strip()
    label, textbody = "page-1", raw
    for i, line in enumerate(raw.splitlines()):
        if line.upper().startswith("LABEL"):
            label = line.split(":", 1)[1].strip() or label
            textbody = "\n".join(raw.splitlines()[i + 1:]).strip()
            break
    slug = _slugify(label)
    return (
        f"<!-- ocr-{slug} -->\n"
        f"OCR Text: {label}\n"
        f"{textbody}\n"
        f"<!-- end ocr-{slug} -->"
    )


@router.post(
    "/image-description",
    response_class=PlainTextResponse,
    summary="Classify and describe image content",
    description=(
        "Upload an image and receive a structured Markdown description based on its automatically "
        "classified content type (Picture, Text/OCR, Table, or Figure). "
        "Each output block is wrapped in HTML comment anchors for reliable LLM chunking.\n\n"
        "**Picture**\n"
        "```\n<!-- picture-{slug} -->\nPicture: {title}\n{description}\n<!-- end picture-{slug} -->\n```\n\n"
        "**OCR Text**\n"
        "```\n<!-- ocr-{slug} -->\nOCR Text: {label}\n{verbatim text}\n<!-- end ocr-{slug} -->\n```\n\n"
        "**Table**\n"
        "```\n<!-- table-{slug} -->\nTable: {title}\n| col | col |\n| --- | --- |\n<!-- end table-{slug} -->\n```\n\n"
        "**Figure**\n"
        "```\n<!-- figure-{slug} -->\nFigure: {title}\nType: {chart-type}\n{description}\n<!-- end figure-{slug} -->\n```\n\n"
        "**Model defaults** (overridable via `.env`):\n\n"
        "| Task | Env var | Default |\n"
        "|-------------|------------------------|----------------------|\n"
        "| Classifier | `LLM_MODEL_CLASSIFIER` | `llama3.2-vision:11b`|\n"
        "| OCR Text | `LLM_MODEL_OCR` | `glm-ocr:bf16` |\n"
        "| Table | `LLM_MODEL_TABLE` | `glm-ocr:bf16` |\n"
        "| Figure | `LLM_MODEL_FIGURE` | `glm-ocr:bf16` |\n"
        "| Picture | `LLM_MODEL_PICTURE` | `llama3.2-vision:11b`|\n"
    ),
    tags=["Utility"],
    response_description="Structured Markdown description wrapped in chunker-friendly HTML comment anchors.",
)
async def image_description(
    file: UploadFile = File(..., description="Image file to classify and describe."),
    user_prompt: Optional[str] = Query(None, description="Optional custom user prompt for Picture descriptions."),
    # --------------------------------------------------------------------------- #
    # Model defaults overridable via .env
    # --------------------------------------------------------------------------- #
    classifier_model: str = Query(
        MODEL_CLASSIFIER,
        description=f"LLM used to classify the image into Picture, Text, Table, or Figure. "
                    f"(Env: LLM_MODEL_CLASSIFIER, default: {MODEL_CLASSIFIER})",
    ),
    ocr_model: str = Query(
        MODEL_OCR,
        description=f"LLM used for OCR Text images. (Env: LLM_MODEL_OCR, default: {MODEL_OCR})",
    ),
    table_model: str = Query(
        MODEL_TABLE,
        description=f"LLM used to reproduce Table images as GFM Markdown. (Env: LLM_MODEL_TABLE, default: {MODEL_TABLE})",
    ),
    figure_model: str = Query(
        MODEL_FIGURE,
        description=f"LLM used to describe Figure/chart images. (Env: LLM_MODEL_FIGURE, default: {MODEL_FIGURE})",
    ),
    picture_model: str = Query(
        MODEL_PICTURE,
        description=f"LLM used to describe Picture/illustration images. (Env: LLM_MODEL_PICTURE, default: {MODEL_PICTURE})",
    ),
    process_types: Optional[List[ImageCategory]] = Query(
        default=None,
        description="Optional filter — only generate descriptions for these categories. "
                    "If omitted, all categories are processed.",
    ),
    temperature: float = Query(
        DEFAULT_TEMPERATURE, ge=0.0, le=1.0,
        description="Sampling temperature for all LLM calls (0.0–1.0).",
    ),
) -> str:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image extension {suffix!r}. Allowed: {sorted(IMAGE_EXTS)}",
        )

    await enforce_max_upload_size(file)

    active_types: List[ImageCategory] = process_types if process_types else list(ImageCategory)
    category_models: Dict[ImageCategory, str] = {
        ImageCategory.text: ocr_model,
        ImageCategory.table: table_model,
        ImageCategory.figure: figure_model,
        ImageCategory.picture: picture_model,
    }

    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)

        logger.info(
            f"image_description file={file.filename!r} "
            f"classifier={classifier_model!r} ocr={ocr_model!r} "
            f"table={table_model!r} figure={figure_model!r} picture={picture_model!r} "
            f"process_types={[t.value for t in active_types]} temperature={temperature}"
        )

        image_b64 = _encode_image_b64(tmp_path)
        category = classify_image(image_b64, classifier_model, temperature)

        if category not in active_types:
            return (
                f"Image classified as {category.value}, which is not in the requested "
                f"{[t.value for t in active_types]}. No description generated."
            )

        description_model = category_models[category]
        dispatch = {
            ImageCategory.picture: _describe_picture,
            ImageCategory.table: _describe_table,
            ImageCategory.figure: _describe_figure,
            ImageCategory.text: _describe_text_ocr,
        }

        if category == ImageCategory.picture:
            return dispatch[category](image_b64, description_model, temperature, user_prompt)

        return dispatch[category](image_b64, description_model, temperature)

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as exc:
                logger.error(f"Failed to clean up temp file {tmp_path}: {exc}")


# ---------------------------------------------------------------------------
# /media_to_text — audio & video transcription via OpenAI Whisper
# ---------------------------------------------------------------------------

# Lazy-loaded Whisper model cache — one model instance per model_size string.
# Using a dict so callers can request different sizes in the same process.
_whisper_models: Dict[str, Any] = {}
_whisper_lock = threading.Lock()


def _get_whisper_model(model_size: str) -> Any:
    """Load (or return cached) a Whisper model by size name."""
    if model_size not in _whisper_models:
        with _whisper_lock:
            if model_size not in _whisper_models:  # double-checked locking
                import whisper as _whisper  # deferred import — only pay cost if endpoint is used
                logger.info(f"Loading Whisper model '{model_size}' — this may take a moment …")
                _whisper_models[model_size] = _whisper.load_model(model_size)
                logger.info(f"Whisper model '{model_size}' ready.")
    return _whisper_models[model_size]


def _transcribe_sync(
    file_path: Path,
    model_size: str,
    language: Optional[str],
    temperature: float,
) -> dict:
    """
    Blocking Whisper transcription — run inside a thread so the event loop stays free.
    Returns the raw Whisper result dict (keys: text, language, segments, …).
    """
    model = _get_whisper_model(model_size)
    options: Dict[str, Any] = {"temperature": temperature}
    if language:
        options["language"] = language
    return model.transcribe(str(file_path), **options)


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@router.post(
    "/media_to_text",
    response_class=PlainTextResponse,
    summary="Transcribe audio or video to text (native language)",
    description=(
        "Upload an audio or video file and receive a plain-text transcription in the file's "
        "**native language** — no translation is performed. "
        "Powered by [OpenAI Whisper](https://github.com/openai/whisper), which supports **99+ languages** "
        "with automatic language detection.\n\n"
        "**Supported audio:** `.mp3` `.wav` `.flac` `.ogg` `.m4a` `.aac` `.wma` `.opus` `.webm`\n\n"
        "**Supported video:** `.mp4` `.mkv` `.avi` `.mov` `.flv` `.wmv` `.ts` `.3gp`\n\n"
        "Audio is extracted from video automatically by Whisper via `ffmpeg`.\n\n"
        "**Output format:**\n"
        "```\n"
        "<!-- media_to_text {filename} language={lang} duration={HH:MM:SS} -->\n"
        "[transcribed text in native language]\n"
        "<!-- end media_to_text {filename-stem} -->\n"
        "```\n\n"
        "**Whisper model sizes** (set via `WHISPER_MODEL_SIZE` env var or the `model_size` query param):\n\n"
        "| Size | ~VRAM | Relative speed | Best for |\n"
        "|----------|--------|----------------|---------------------------------|\n"
        "| tiny | ~1 GB | ~32× | Quick drafts, low-resource hosts |\n"
        "| base | ~1 GB | ~16× | Good accuracy, fast (default) |\n"
        "| small | ~2 GB | ~6× | Balanced accuracy/speed |\n"
        "| medium | ~5 GB | ~2× | High accuracy |\n"
        "| large-v3 | ~10 GB | 1× | Best accuracy, all languages |\n\n"
        "> **System requirement:** `ffmpeg` must be installed and on `$PATH`.\n"
        "> Install with `sudo apt install ffmpeg` (Debian/Ubuntu) or `brew install ffmpeg` (macOS).\n"
    ),
    tags=["Utility"],
    response_description=(
        "Plain-text transcription in the file's native language, "
        "wrapped in HTML comment anchors that carry metadata (filename, detected language, duration)."
    ),
)
async def media_to_text(
    file: UploadFile = File(..., description="Audio or video file to transcribe."),
    model_size: str = Query(
        WHISPER_MODEL_SIZE,
        description=(
            "Whisper model size: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`. "
            f"(Env: `WHISPER_MODEL_SIZE`, current default: `{WHISPER_MODEL_SIZE}`)"
        ),
    ),
    language: Optional[str] = Query(
        None,
        description=(
            "BCP-47 language code to force, e.g. `en`, `fr`, `ja`, `ar`, `zh`, `de`. "
            "If omitted, Whisper auto-detects the spoken language. "
            "Forcing the language can improve accuracy and speed."
        ),
    ),
    temperature: float = Query(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Whisper decoding temperature. `0.0` = greedy/deterministic (recommended). "
            "Higher values increase randomness and may help with hard audio."
        ),
    ),
) -> str:
    # ── validate extension ──────────────────────────────────────────────────
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in MEDIA_EXTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported media extension {suffix!r}. "
                f"Allowed audio: {sorted(AUDIO_EXTS)}. "
                f"Allowed video: {sorted(VIDEO_EXTS)}."
            ),
        )

    await enforce_max_upload_size(file)

    tmp_path: Optional[Path] = None
    try:
        # ── save upload to temp file (preserving extension so ffmpeg/Whisper recognises it) ──
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = await file.read(1024 * 1024)  # stream 1 MB at a time
                if not chunk:
                    break
                tmp.write(chunk)

        logger.info(
            f"media_to_text file={file.filename!r} size={tmp_path.stat().st_size} "
            f"model_size={model_size!r} language={language!r} temperature={temperature}"
        )

        # ── run blocking Whisper call in a thread pool ──────────────────────
        result: dict = await asyncio.to_thread(
            _transcribe_sync, tmp_path, model_size, language, temperature
        )

        detected_lang: str = result.get("language", "unknown")
        text: str = (result.get("text") or "").strip()
        segments: list = result.get("segments") or []
        duration_sec = float(segments[-1]["end"]) if segments else 0.0
        duration_str = _fmt_duration(duration_sec)
        slug = _slugify(Path(file.filename or "media").stem)

        logger.info(
            f"media_to_text done — language={detected_lang} duration={duration_str} "
            f"chars={len(text)}"
        )

        return (
            f"<!-- media_to_text {file.filename} language={detected_lang} duration={duration_str} -->\n"
            f"{text}\n"
            f"<!-- end media_to_text {slug} -->"
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"media_to_text transcription failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception as exc:
                logger.error(f"Failed to clean up temp file {tmp_path}: {exc}")


# ---------------------------------------------------------------------------
# /general_llm_chat — generic text chat via LLM /api/chat
# ---------------------------------------------------------------------------

GENERAL_LLM_SYSTEM_PROMPT = os.getenv(
    "GENERAL_LLM_SYSTEM_PROMPT",
    "You are a helpful, accurate, and concise AI assistant.",
)

GENERAL_LLM_USER_PROMPT = os.getenv("GENERAL_LLM_USER_PROMPT", " ")
GENERAL_LLM_TEMPERATURE = float(os.getenv("GENERAL_LLM_TEMPERATURE", "0.1"))
GENERAL_LLM_MAX_TOKENS = int(os.getenv("GENERAL_LLM_MAX_TOKENS", "4096"))
GENERAL_LLM_MODEL = os.getenv("GENERAL_LLM_MODEL", "gemma3:12b")


class GeneralLLMChatResponse(BaseModel):
    model: str = Field(..., description="Model used for the response.")
    created_at: Optional[str] = Field(None, description="Timestamp returned by LLM.")
    done: Optional[bool] = Field(None, description="Whether generation completed.")
    done_reason: Optional[str] = Field(None, description="Why generation stopped.")
    total_duration: Optional[int] = Field(None, description="Total request duration in nanoseconds.")
    load_duration: Optional[int] = Field(None, description="Model load duration in nanoseconds.")
    prompt_eval_count: Optional[int] = Field(None, description="Prompt token count.")
    prompt_eval_duration: Optional[int] = Field(None, description="Prompt evaluation duration in nanoseconds.")
    eval_count: Optional[int] = Field(None, description="Generated token count.")
    eval_duration: Optional[int] = Field(None, description="Generation duration in nanoseconds.")
    response: str = Field(..., description="Assistant text response.")


def _call_general_llm_chat(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_token: int,
    model: str,
    base_url: str = LLM_MODEL_CHAT_BASE_URL,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": max_token,
    }
    logger.info(
        f"Calling general LLM chat url={url} model={model!r} "
        f"temperature={temperature} max_token={max_token}"
    )
    try:
        resp = _sync_http_client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()

        choice = (data.get("choices") or [{}])[0]
        return {
            "model": data.get("model", model),
            "created_at": data.get("created_at"),
            "done": data.get("done"),
            "done_reason": choice.get("finish_reason"),
            "total_duration": data.get("total_duration"),
            "load_duration": data.get("load_duration"),
            "prompt_eval_count": (data.get("usage") or {}).get("prompt_tokens"),
            "prompt_eval_duration": data.get("prompt_eval_duration"),
            "eval_count": (data.get("usage") or {}).get("completion_tokens"),
            "eval_duration": data.get("eval_duration"),
            "response": (choice.get("message") or {}).get("content", "").strip(),
        }
    except Exception as exc:
        logger.exception(f"general_llm_chat failed: {exc}")
        raise HTTPException(status_code=500, detail=f"LLM chat failed: {exc}") from exc


@router.post(
    "/general_llm_chat",
    summary="Generic LLM chat endpoint",
    description=(
        "Send a system prompt and user prompt to an compatible chat model "
        "using `POST /v1/chat/completions` and receive a structured JSON response.\n\n"
        "**Defaults**\n"
        "- `system_prompt`: `You are a helpful, accurate, and concise AI assistant.`\n"
        "- `user_prompt`: ` `\n"
        "- `temperature`: `0.1`\n"
        "- `max_token`: `4096`\n"
        "- `model`: `gemma3:12b`\n\n"
        "**Download option**\n"
        "- Set `download=true` to receive the same JSON as a downloadable `.json` file."
    ),
    tags=["Utility"],
    response_model=GeneralLLMChatResponse,
    response_description="Structured JSON response from the LLM chat call.",
)
async def general_llm_chat(
    system_prompt: str = Query(
        GENERAL_LLM_SYSTEM_PROMPT,
        description="System prompt sent to the LLM.",
    ),
    user_prompt: str = Query(
        GENERAL_LLM_USER_PROMPT,
        description="User prompt sent to the LLM.",
    ),
    temperature: float = Query(
        GENERAL_LLM_TEMPERATURE,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (0.0 to 1.0).",
    ),
    max_token: int = Query(
        GENERAL_LLM_MAX_TOKENS,
        ge=1,
        description="Maximum number of tokens to generate.",
    ),
    model: str = Query(
        GENERAL_LLM_MODEL,
        description="LLM model name.",
    ),
    download: bool = Query(
        False,
        description="If true, returns the response as a downloadable JSON file.",
    ),
    base_url: str = Query(
        LLM_MODEL_CHAT_BASE_URL,
        description="LLM-compatible chat base URL.",
    ),
):
    result = _call_general_llm_chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_token=max_token,
        model=model,
        base_url=base_url,
    )

    if download:
        json_bytes = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
        filename = f"general_llm_chat_{_slugify(model)}.json"
        return StreamingResponse(
            BytesIO(json_bytes),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return JSONResponse(content=result)
