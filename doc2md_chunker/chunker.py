# doc2md_chunker/chunker.py
import os
import logging
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from pydantic import BaseModel, validator
from typing import List, Optional, Literal, Any, Sequence
import httpx
from pathlib import Path
from io import BytesIO
from fastapi.concurrency import run_in_threadpool
from security import get_api_key

from chonkie import TokenChunker, SentenceChunker, RecursiveChunker, SemanticChunker

try:
    from chonkie import SDPMChunker
    _HAS_SDPM_CHUNKER = True
except ImportError:
    _HAS_SDPM_CHUNKER = False

try:
    from chonkie import LateChunker
    _HAS_LATE_CHUNKER = True
except ImportError:
    _HAS_LATE_CHUNKER = False

try:
    from chonkie import NeuralChunker
    _HAS_NEURAL_CHUNKER = True
except ImportError:
    _HAS_NEURAL_CHUNKER = False

try:
    from chonkie import CodeChunker
    _HAS_CODE_CHUNKER = True
except ImportError:
    _HAS_CODE_CHUNKER = False

try:
    from chonkie import SlumberChunker
    _HAS_SLUMBER_CHUNKER = True
except ImportError:
    _HAS_SLUMBER_CHUNKER = False

try:
    from chonkie import TableChunker
    _HAS_TABLE_CHUNKER = True
except ImportError:
    _HAS_TABLE_CHUNKER = False

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parent.parent / "doc2md_chunker.env"
load_dotenv(ENV_PATH)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model path resolution (airgap-safe: prefers local copy over HF Hub)
# ---------------------------------------------------------------------------
STATIC_MODEL_ROOT = os.getenv("DOC2MD_STATIC_MODEL_ROOT", "/models/huggingface")


def _resolve_model_path(model_name: str) -> str:
    """Return the local filesystem path for a HuggingFace model if it exists,
    otherwise return the hub repo-id so Chonkie can attempt an online download."""
    if model_name.startswith("/"):
        return model_name
    folder = model_name.replace("/", "--")
    local = Path(STATIC_MODEL_ROOT) / folder
    if local.exists():
        return str(local)
    # Fallback: let Chonkie/transformers try the hub (will fail in true airgap)
    return model_name


_SEMANTIC_MODEL_DEFAULT = _resolve_model_path(
    os.getenv("DOC2MD_SEMANTIC_CHUNKER_MODEL", "minishlab/potion-base-32M")
)
_SDPM_MODEL_DEFAULT = _resolve_model_path(
    os.getenv("DOC2MD_SDPM_CHUNKER_MODEL", "minishlab/potion-base-32M")
)
_LATE_MODEL_DEFAULT = _resolve_model_path(
    os.getenv("DOC2MD_LATE_CHUNKER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
)
_NEURAL_MODEL_DEFAULT = _resolve_model_path(
    os.getenv("DOC2MD_NEURAL_CHUNKER_MODEL", "mirth/chonky_modernbert_large_1")
)

# ---------------------------------------------------------------------------
# HTTP / LLM config
# ---------------------------------------------------------------------------
HTTPX_TIMEOUT = httpx.Timeout(float(os.getenv("DOC2MD_HTTPX_TIMEOUT", "600.0")))

LLM_MODEL_DEFAULT_BASE_URL = os.getenv(
    "DOC2MD_LLM_MODEL_DEFAULT_BASE_URL", "http://localhost:11434"
).rstrip("/")
LLM_MODEL_CHAT_BASE_URL    = os.getenv("DOC2MD_LLM_MODEL_CHAT_BASE_URL",    LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")
LLM_MODEL_TRANSLATE_BASE_URL = os.getenv("DOC2MD_LLM_MODEL_TRANSLATE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")
LLM_MODEL_CHUNK_BASE_URL   = os.getenv("DOC2MD_LLM_MODEL_CHUNK_BASE_URL",   LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")
LLM_MODEL_CLASSIFIER_BASE_URL = os.getenv("DOC2MD_LLM_MODEL_CLASSIFIER_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")
LLM_MODEL_TABLE_BASE_URL   = os.getenv("DOC2MD_LLM_MODEL_TABLE_BASE_URL",   LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")
LLM_MODEL_OCR_BASE_URL     = os.getenv("DOC2MD_LLM_MODEL_OCR_BASE_URL",     LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")
LLM_MODEL_PICTURE_BASE_URL = os.getenv("DOC2MD_LLM_MODEL_PICTURE_BASE_URL", LLM_MODEL_DEFAULT_BASE_URL).rstrip("/")

ALLOWED_ORIGINS = os.getenv("DOC2MD_ALLOWED_ORIGINS", "").split(",")

# ---------------------------------------------------------------------------
# SlumberChunker LLM backend
# ---------------------------------------------------------------------------
_SLUMBER_BACKEND       = os.getenv("DOC2MD_LLM_SLUMBER_BACKEND", "ollama").lower()
_SLUMBER_OLLAMA_MODEL  = os.getenv("DOC2MD_LLM_SLUMBER_OLLAMA_MODEL", "gemma3:4b")
_SLUMBER_LOCAL_MODEL   = _resolve_model_path(
    os.getenv("DOC2MD_LLM_SLUMBER_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
)
_SLUMBER_LOCAL_MAX_NEW = int(os.getenv("DOC2MD_LLM_SLUMBER_LOCAL_MAX_NEW_TOKENS", "512"))
_SLUMBER_LOCAL_TEMP    = float(os.getenv("DOC2MD_LLM_SLUMBER_LOCAL_TEMPERATURE", "0.1"))

# Lazy-loaded local pipeline (only initialised when backend=="local")
_local_pipeline = None

def _get_local_pipeline():
    """Lazily load a HuggingFace text-generation pipeline for SlumberChunker."""
    global _local_pipeline
    if _local_pipeline is None:
        try:
            from transformers import pipeline as hf_pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1
            log.info("Loading local SlumberChunker LLM from %s (device=%s)", _SLUMBER_LOCAL_MODEL, device)
            _local_pipeline = hf_pipeline(
                "text-generation",
                model=_SLUMBER_LOCAL_MODEL,
                device=device,
                torch_dtype="auto",
                trust_remote_code=True,
            )
            log.info("Local SlumberChunker LLM loaded.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local LLM for SlumberChunker from '{_SLUMBER_LOCAL_MODEL}': {exc}"
            ) from exc
    return _local_pipeline


def _slumber_ollama_fn(prompt: str) -> str:
    """Synchronous Ollama/OpenAI-compatible call used as the llm_fn for SlumberChunker.
    Uses a plain httpx sync client — no event-loop conflicts inside uvicorn threads."""
    url = f"{LLM_MODEL_CHUNK_BASE_URL}/v1/chat/completions"
    payload = {
        "model": _SLUMBER_OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    with httpx.Client(timeout=HTTPX_TIMEOUT) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _slumber_local_fn(prompt: str) -> str:
    """Synchronous local-HF call used as the llm_fn for SlumberChunker.
    Loads the pipeline lazily on first call."""
    pipe = _get_local_pipeline()
    results = pipe(
        prompt,
        max_new_tokens=_SLUMBER_LOCAL_MAX_NEW,
        temperature=_SLUMBER_LOCAL_TEMP,
        do_sample=(_SLUMBER_LOCAL_TEMP > 0),
        return_full_text=False,
    )
    return results[0]["generated_text"]


def _get_slumber_llm_fn():
    """Return the correct llm_fn based on LLM_SLUMBER_BACKEND env var."""
    if _SLUMBER_BACKEND == "local":
        return _slumber_local_fn
    # default: ollama
    return _slumber_ollama_fn


# ---------------------------------------------------------------------------
# Shared async HTTP client
# ---------------------------------------------------------------------------
asynchttpclient = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)

router = APIRouter(
    tags=["chunker"],
    dependencies=[Depends(get_api_key)],
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class ChunkMetadata(BaseModel):
    chunk_id: int
    text: str
    document_name: str
    page: int
    start: int
    end: int
    token_count: Optional[int] = None
    chunker_type: str


class ChunkMetadataResponse(BaseModel):
    chunks: List[ChunkMetadata]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _read_upload_stream(file: UploadFile, chunk_size: int = 1024 * 1024) -> str:
    buf = BytesIO()
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        buf.write(chunk)
    return buf.getvalue().decode("utf-8", errors="replace")


def build_chunks_response(
    chunks,
    document_name: str,
    chunker_type: str,
    page: int = 1,
) -> ChunkMetadataResponse:
    items: List[ChunkMetadata] = []
    offset = 0
    for idx, c in enumerate(chunks):
        text = c.text if hasattr(c, "text") else str(c)
        length = len(text)
        token_count = getattr(c, "token_count", None)
        items.append(
            ChunkMetadata(
                chunk_id=idx,
                text=text,
                document_name=document_name,
                page=page,
                start=offset,
                end=offset + length,
                token_count=token_count,
                chunker_type=chunker_type,
            )
        )
        offset += length
    return ChunkMetadataResponse(chunks=items)


# ---------------------------------------------------------------------------
# Token Chunker
# ---------------------------------------------------------------------------
@router.post("/token", response_model=ChunkMetadataResponse)
async def chunk_token(
    file: UploadFile = File(...),
    chunk_size: int = Query(512),
    chunk_overlap: int = Query(0),
    tokenizer: str = Query("gpt2"),
):
    """Fixed-size token/word chunker. Set tokenizer='word' for plain word-count mode."""
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    chunker = TokenChunker(tokenizer=tokenizer, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk(text)
    chunker_type = "word" if tokenizer == "word" else "token"
    return build_chunks_response(chunks, document_name=file.filename, chunker_type=chunker_type)


# ---------------------------------------------------------------------------
# Sentence Chunker
# ---------------------------------------------------------------------------
@router.post("/sentence", response_model=ChunkMetadataResponse)
async def chunk_sentence(
    file: UploadFile = File(...),
    max_characters: int = Query(2000),
):
    """Sentence-boundary-aware chunker."""
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    chunker = SentenceChunker(tokenizer="character", chunk_size=max_characters)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="sentence")


# ---------------------------------------------------------------------------
# Recursive Chunker
# ---------------------------------------------------------------------------
@router.post("/recursive", response_model=ChunkMetadataResponse)
async def chunk_recursive(
    file: UploadFile = File(...),
    max_characters: int = Query(2000),
):
    """Hierarchical delimiter chunker (paragraph → sentence → word)."""
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    chunker = RecursiveChunker(tokenizer="character", chunk_size=max_characters)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="recursive")


# ---------------------------------------------------------------------------
# Semantic Chunker  (embedding model — runs entirely inside doc2md)
# ---------------------------------------------------------------------------
@router.post("/semantic", response_model=ChunkMetadataResponse)
async def chunk_semantic(
    file: UploadFile = File(...),
    embedding_model: str = Query(
        "minishlab/potion-base-32M",
        description=(
            "HuggingFace embedding model for semantic similarity. "
            "Will be resolved to /models/huggingface/<org>--<name> for airgap use. "
            "Examples: 'minishlab/potion-base-32M', 'sentence-transformers/all-MiniLM-L6-v2'."
        ),
    ),
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    chunk_size: int = Query(512),
):
    """
    Embedding-based semantic chunker. Groups text by topical similarity using a
    local HuggingFace model — no Ollama or internet required.

    Models are resolved from STATIC_MODEL_ROOT (/models/huggingface) automatically.
    Pre-download with: python download_models.py
    """
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    resolved = _resolve_model_path(embedding_model)
    chunker = SemanticChunker(
        embedding_model=resolved,
        threshold=threshold,
        chunk_size=chunk_size,
    )
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="semantic")


# ---------------------------------------------------------------------------
# SDPM Chunker  (embedding model — runs entirely inside doc2md)
# ---------------------------------------------------------------------------
@router.post("/sdpm", response_model=ChunkMetadataResponse)
async def chunk_sdpm(
    file: UploadFile = File(...),
    embedding_model: str = Query("minishlab/potion-base-32M"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    chunk_size: int = Query(512),
):
    """
    Semantic Double-Pass Merge chunker. Two embedding passes produce cleaner
    topic boundaries than single-pass semantic chunking.

    Runs entirely inside doc2md using local HF models.
    """
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not _HAS_SDPM_CHUNKER:
        raise HTTPException(status_code=501, detail="SDPMChunker not available. pip install --upgrade chonkie")

    resolved = _resolve_model_path(embedding_model)
    chunker = SDPMChunker(embedding_model=resolved, threshold=threshold, chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="sdpm")


# ---------------------------------------------------------------------------
# Late Chunker  (long-context encoder — runs entirely inside doc2md)
# ---------------------------------------------------------------------------
@router.post("/late", response_model=ChunkMetadataResponse)
async def chunk_late(
    file: UploadFile = File(...),
    embedding_model: str = Query(
        "sentence-transformers/all-MiniLM-L6-v2",
        description=(
            "Long-context encoder for late chunking. For best results use a model "
            "with ≥2048-token context, e.g. 'jinaai/jina-embeddings-v2-base-en' (8192 ctx) "
            "or 'nomic-ai/nomic-embed-text-v1' (8192 ctx). "
            "Falls back to 'sentence-transformers/all-MiniLM-L6-v2' (512 ctx, less effective). "
            "Model is resolved from /models/huggingface for airgap use."
        ),
    ),
    chunk_size: int = Query(512),
):
    """
    Late Chunking: embeds the FULL document first (preserving cross-chunk context),
    then applies boundaries — avoids the information loss of naive split-then-embed.

    Runs entirely inside doc2md using a local HF encoder. No Ollama needed.

    Best models for airgap pre-download:
      • jinaai/jina-embeddings-v2-base-en   (8192 ctx, 137M, best quality)
      • nomic-ai/nomic-embed-text-v1        (8192 ctx, 137M)
      • sentence-transformers/all-MiniLM-L6-v2  (512 ctx, 22M, lightest)
    """
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not _HAS_LATE_CHUNKER:
        raise HTTPException(status_code=501, detail="LateChunker not available. pip install --upgrade chonkie")

    resolved = _resolve_model_path(embedding_model)
    chunker = LateChunker(embedding_model=resolved, chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="late")


# ---------------------------------------------------------------------------
# Neural Chunker  (ModernBERT boundary model — runs entirely inside doc2md)
# ---------------------------------------------------------------------------
@router.post("/neural", response_model=ChunkMetadataResponse)
async def chunk_neural(
    file: UploadFile = File(...),
    model: str = Query(
        "mirth/chonky_modernbert_large_1",
        description="Neural boundary-detection model. Resolved from /models/huggingface for airgap.",
    ),
    chunk_size: int = Query(512),
):
    """
    Neural boundary-detection chunker trained to predict split points in prose.
    Uses a local HF model (ModernBERT) — no Ollama needed.
    """
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not _HAS_NEURAL_CHUNKER:
        raise HTTPException(status_code=501, detail="NeuralChunker not available. pip install --upgrade chonkie")

    resolved = _resolve_model_path(model)
    chunker = NeuralChunker(model=resolved, chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="neural")


# ---------------------------------------------------------------------------
# Code Chunker
# ---------------------------------------------------------------------------
@router.post("/code", response_model=ChunkMetadataResponse)
async def chunk_code(
    file: UploadFile = File(...),
    language: str = Query("python"),
    chunk_size: int = Query(512),
):
    """Syntax-aware source-code chunker (tree-sitter). No model required."""
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not _HAS_CODE_CHUNKER:
        raise HTTPException(status_code=501, detail="CodeChunker not available. pip install --upgrade chonkie")

    chunker = CodeChunker(language=language, chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="code")


# ---------------------------------------------------------------------------
# Table Chunker
# ---------------------------------------------------------------------------
@router.post("/table", response_model=ChunkMetadataResponse)
async def chunk_table(
    file: UploadFile = File(...),
    chunk_size: int = Query(3),
    tokenizer: str = Query("row"),
):
    """Markdown/HTML table chunker — preserves header in every chunk. No model required."""
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not _HAS_TABLE_CHUNKER:
        raise HTTPException(status_code=501, detail="TableChunker not available. pip install --upgrade chonkie")

    chunker = TableChunker(tokenizer=tokenizer, chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return build_chunks_response(chunks, document_name=file.filename, chunker_type="table")


# ---------------------------------------------------------------------------
# LLM Slumber Chunker  (Agentic — uses LLM for boundary decisions)
# ---------------------------------------------------------------------------
@router.post("/llm-slumber", response_model=ChunkMetadataResponse)
async def chunk_llm_slumber(
    file: UploadFile = File(...),
    model: str = Query(
        None,
        description=(
            "Override the LLM model. "
            "For 'ollama' backend: Ollama model name (e.g. 'gemma3:4b', 'llama3.1'). "
            "For 'local' backend: HF model id (e.g. 'Qwen/Qwen2.5-1.5B-Instruct'). "
            "Leave blank to use LLM_SLUMBER_OLLAMA_MODEL / LLM_SLUMBER_LOCAL_MODEL from env."
        ),
    ),
    backend: str = Query(
        None,
        description=(
            "LLM backend: 'ollama' (default) calls the Ollama container over the "
            "Docker-internal network — airgap safe. "
            "'local' loads a HuggingFace causal-LM directly inside this container "
            "— fully self-contained but requires VRAM and LLM_SLUMBER_LOCAL_MODEL "
            "pre-downloaded under STATIC_MODEL_ROOT."
        ),
    ),
    max_tokens: int = Query(256, description="Max tokens for the LLM chunking decision."),
):
    """
    Agentic LLM-driven chunker. The LLM reads each text segment and decides
    where meaningful boundaries lie — the highest quality chunker for nuanced content.

    ## Backends

    | Backend  | Where LLM runs          | Airgap | VRAM cost in doc2md |
    |----------|--------------------------|--------|----------------------|
    | `ollama` | Ollama container (network)| ✅    | None (offloaded)     |
    | `local`  | Inside this container    | ✅    | ~3–8 GB              |

    **ollama** is the default and recommended approach — the Ollama container is
    already on the same Docker-internal network and already loaded with models.

    **local** is useful when you want doc2md to be fully independent of Ollama,
    but it competes for VRAM with OCR and classification models.
    Pre-download the model with: `python download_models.py --slumber-local`
    """
    try:
        text = await _read_upload_stream(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    if not _HAS_SLUMBER_CHUNKER:
        raise HTTPException(
            status_code=501,
            detail="SlumberChunker not available. pip install --upgrade chonkie",
        )

    # Determine backend and llm_fn
    effective_backend = (backend or _SLUMBER_BACKEND).lower()

    if effective_backend == "local":
        # Override model if provided via query param
        global _SLUMBER_LOCAL_MODEL
        if model:
            _SLUMBER_LOCAL_MODEL = _resolve_model_path(model)
        llm_fn = _slumber_local_fn
        chunker_type_label = "slumber-local"
    else:
        # Ollama backend — honour per-request model override
        effective_ollama_model = model or _SLUMBER_OLLAMA_MODEL

        def llm_fn(prompt: str) -> str:  # closure over effective_ollama_model
            url = f"{LLM_MODEL_CHUNK_BASE_URL}/v1/chat/completions"
            payload = {
                "model": effective_ollama_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            with httpx.Client(timeout=HTTPX_TIMEOUT) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        chunker_type_label = f"slumber-ollama-{effective_ollama_model}"

    try:
        chunker = SlumberChunker(llm_fn=llm_fn)
        chunks = await run_in_threadpool(chunker.chunk, text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"SlumberChunker error: {exc}")

    return build_chunks_response(chunks, document_name=file.filename, chunker_type=chunker_type_label)


# ---------------------------------------------------------------------------
# Generic unified endpoint
# ---------------------------------------------------------------------------
ChunkerName = Literal[
    "token", "sentence", "recursive",
    "semantic", "sdpm", "late", "neural",
    "code", "table",
]

_CHUNKER_REGISTRY: dict[str, Any] = {
    "token":     TokenChunker,
    "sentence":  SentenceChunker,
    "recursive": RecursiveChunker,
    "semantic":  SemanticChunker,
}
if _HAS_SDPM_CHUNKER:
    _CHUNKER_REGISTRY["sdpm"]   = SDPMChunker
if _HAS_LATE_CHUNKER:
    _CHUNKER_REGISTRY["late"]   = LateChunker
if _HAS_NEURAL_CHUNKER:
    _CHUNKER_REGISTRY["neural"] = NeuralChunker
if _HAS_CODE_CHUNKER:
    _CHUNKER_REGISTRY["code"]   = CodeChunker
if _HAS_TABLE_CHUNKER:
    _CHUNKER_REGISTRY["table"]  = TableChunker


def get_available_chunkers() -> Sequence[ChunkerName]:
    return list(_CHUNKER_REGISTRY.keys())


def chunk_text(text: str, method: ChunkerName = "recursive", **chunker_kwargs: Any):
    if method not in _CHUNKER_REGISTRY:
        raise ValueError(
            f"Unknown chunking method '{method}'. Supported: {', '.join(get_available_chunkers())}"
        )
    # Resolve embedding_model / model kwargs for airgap
    for key in ("embedding_model", "model"):
        if key in chunker_kwargs and isinstance(chunker_kwargs[key], str):
            chunker_kwargs[key] = _resolve_model_path(chunker_kwargs[key])

    chunker_cls = _CHUNKER_REGISTRY[method]
    chunker = chunker_cls(**chunker_kwargs)
    return chunker(text)


class ChunkWithDescription(BaseModel):
    text: str
    index: int
    token_count: Optional[int] = None
    description: str


class ChunkResponse(BaseModel):
    chunks: List[ChunkWithDescription]


class ChunkRequest(BaseModel):
    text: str
    method: ChunkerName = "recursive"
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    max_tokens: Optional[int] = None
    separators: Optional[List[str]] = None
    similarity_threshold: Optional[float] = None
    min_chunk_size: Optional[int] = None
    max_chunk_size: Optional[int] = None
    language: Optional[str] = None
    model: Optional[str] = None
    tokenizer: Optional[str] = None
    embedding_model: Optional[str] = None

    @validator("chunk_size")
    def validate_chunk_size(cls, v):
        if v is not None and v <= 0:
            raise ValueError("chunk_size must be > 0")
        return v

    @validator("similarity_threshold")
    def validate_similarity_threshold(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("similarity_threshold must be in [0.0, 1.0]")
        return v


@router.get("/chunkers-list", response_model=List[ChunkerName])
async def api_get_chunkers():
    """Return all available chunker method names (runtime discovery)."""
    return list(get_available_chunkers())


@router.post("/chunk", response_model=ChunkResponse)
async def api_chunk_text(req: ChunkRequest):
    """Chunk text using any supported Chonkie strategy (JSON body).
    Note: for LLM-guided SlumberChunker use /llm-slumber."""
    raw_kwargs: dict[str, Any] = dict(
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
        max_tokens=req.max_tokens,
        separators=req.separators,
        similarity_threshold=req.similarity_threshold,
        min_chunk_size=req.min_chunk_size,
        max_chunk_size=req.max_chunk_size,
        language=req.language,
        model=req.model,
        tokenizer=req.tokenizer,
        embedding_model=req.embedding_model,
    )
    kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}

    try:
        chunks = await run_in_threadpool(chunk_text, req.text, method=req.method, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Chunker parameter error for '{req.method}': {e}")

    payload: List[ChunkWithDescription] = []
    for idx, ch in enumerate(chunks):
        ch_text   = getattr(ch, "text", str(ch))
        ch_tokens = getattr(ch, "token_count", None)
        ch_index  = getattr(ch, "index", idx)
        desc_parts = [f"Chunk {ch_index}"]
        desc_parts.append(f"{ch_tokens} tokens" if ch_tokens is not None else f"{len(ch_text)} characters")
        payload.append(
            ChunkWithDescription(
                text=ch_text,
                index=ch_index,
                token_count=ch_tokens,
                description=", ".join(desc_parts),
            )
        )
    return ChunkResponse(chunks=payload)


@router.post("/chunk-file", response_model=ChunkResponse)
async def api_chunk_file(
    file: UploadFile = File(...),
    method: ChunkerName = Query("recursive"),
    chunk_size: Optional[int] = Query(None),
    chunk_overlap: Optional[int] = Query(None),
    max_tokens: Optional[int] = Query(None),
    separators: Optional[List[str]] = Query(None),
    similarity_threshold: Optional[float] = Query(None),
    min_chunk_size: Optional[int] = Query(None),
    max_chunk_size: Optional[int] = Query(None),
    language: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    tokenizer: Optional[str] = Query(None),
    embedding_model: Optional[str] = Query(None),
):
    """Upload a file and chunk it using any supported Chonkie strategy.
    For LLM-guided SlumberChunker use /llm-slumber."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".md", ".markdown", ".txt", ".py", ".ts", ".js", ".java", ".go", ".rs", ".html", ".csv"}:
        raise HTTPException(status_code=400, detail=f"Unsupported extension '{suffix}'.")

    try:
        data = await file.read()
        text = data.decode("utf-8", errors="replace")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    raw_kwargs: dict[str, Any] = dict(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, max_tokens=max_tokens,
        separators=separators, similarity_threshold=similarity_threshold,
        min_chunk_size=min_chunk_size, max_chunk_size=max_chunk_size,
        language=language, model=model, tokenizer=tokenizer, embedding_model=embedding_model,
    )
    kwargs = {k: v for k, v in raw_kwargs.items() if v is not None}

    try:
        chunks = chunk_text(text, method=method, **kwargs)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Chunker parameter error for '{method}': {e}")

    payload: List[ChunkWithDescription] = []
    for idx, ch in enumerate(chunks):
        ch_text   = getattr(ch, "text", str(ch))
        ch_tokens = getattr(ch, "token_count", None)
        ch_index  = getattr(ch, "index", idx)
        desc_parts = [f"Chunk {ch_index}"]
        desc_parts.append(f"{ch_tokens} tokens" if ch_tokens is not None else f"{len(ch_text)} characters")
        payload.append(
            ChunkWithDescription(
                text=ch_text, index=ch_index, token_count=ch_tokens,
                description=", ".join(desc_parts),
            )
        )
    return ChunkResponse(chunks=payload)
