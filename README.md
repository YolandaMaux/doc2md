
## Recommended Model Stack

| Task | Model | Params | Quant | Approx. VRAM | Why |
|---|---|---|---|---|---|
| Image classifier | Qwen3-VL-8B-Instruct-GGUF | 8B | Q4_K_M | ~5–6GB text + ~1GB mmproj | Same model as describer below — zero-shot classify via prompt instead of loading a second model  [huggingface](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| Multilingual OCR | PaddleOCR-VL-GGUF | 0.9B | Q4/Q8 | ~0.6–1GB | Purpose-built for 100+ language OCR, most battle-tested multilingual pipeline in llama.cpp  [local-ai-zone.github](https://local-ai-zone.github.io/guides/best-ai-ocr-models-ultimate-ranking-2026.html) |
| Image describer | Qwen3-VL-8B-Instruct-GGUF | 8B | Q4_K_M | ~6–7GB total | Strong general captioning/detailed description, native llama.cpp GGUF support via bartowski/ggml-org builds  [huggingface](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF) |
| Figure/table/plot describer | GLM-OCR-GGUF | 0.9B | Q4/Q8 | ~0.6–1GB | 2026's standout tiny model specifically for complex tables, charts, and formula recognition  [local-ai-zone.github](https://local-ai-zone.github.io/blog/top-embedding-reranker-ocr-models-2026.html) |

## Two Viable Configurations

**Consolidated (recommended):** Run Qwen3-VL-8B-Instruct as your one workhorse for both classification and general description (it handles both via prompting — Qwen3-VL is explicitly benchmarked as strong across document/chart/general vision tasks), then keep PaddleOCR-VL and GLM-OCR loaded simultaneously as tiny specialists. Total: ~7GB (Qwen3-VL) + ~1GB (PaddleOCR-VL) + ~1GB (GLM-OCR) ≈ 9GB, leaving ~3GB headroom for KV cache/context on a 12GB card. [huggingface](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)

**Fully specialized alternative:** If you want a true classifier separate from your describer, swap Qwen3-VL-8B for Qwen3-VL-2B-Instruct (~1.5GB at Q4) for classification duty, and add a dedicated dense-document specialist like DeepSeek-OCR-GGUF (3B, ~2–3GB) for figure/table parsing instead of GLM-OCR if you need markdown-quality table extraction over pure description. Note DeepSeek-OCR currently needs an unmerged llama.cpp PR (`pr24975`) rather than mainline support. [local-ai-zone.github](https://local-ai-zone.github.io/blog/top-embedding-reranker-ocr-models-2026.html)

## Practical Notes

- Every vision GGUF needs two files: the language-model quant and a separate `mmproj` (vision projector) file, almost always kept at F16 regardless of the LLM's quant level — budget for that extra ~0.5–1.5GB per model. [localaimaster](https://localaimaster.com/blog/qwen-3-vl-local-setup)
- PaddleOCR-VL and GLM-OCR are both sub-1B and integrated into mainline llama.cpp, making them the safest low-VRAM specialists rather than repurposing a general VLM for OCR/table work. [local-ai-zone.github](https://local-ai-zone.github.io/guides/best-ai-ocr-models-ultimate-ranking-2026.html)
- Given your existing stack already runs Qwen3-Embedding-4B and a Qwen3-reranker concurrently for RAG, factor that VRAM in too — if those stay resident alongside vision models, lean toward the consolidated config or load vision models on-demand per pipeline stage rather than keeping all four resident at once.



## Vision Models/OCR


### OmniDocBench v1.5 Head-to-Head

| Metric | PaddleOCR-VL-1.5 | GLM-OCR | Edge |
|---|---|---|---|
| Overall score | 94.50 | 94.62 | GLM-OCR (slim)  [arxiv](https://arxiv.org/html/2603.10910v2) |
| Text edit distance (lower=better) | 0.035 | 0.040 | PaddleOCR-VL  [arxiv](https://arxiv.org/html/2603.10910v2) |
| Formula (CDM) | 94.21 | 93.90 | PaddleOCR-VL  [arxiv](https://arxiv.org/html/2603.10910v2) |
| Table (TEDS) | 92.76 | 93.96 | GLM-OCR  [arxiv](https://arxiv.org/html/2603.10910v2) |
| Table (TEDS-S) | 95.79 | 96.39 | GLM-OCR  [arxiv](https://arxiv.org/html/2603.10910v2) |
| Reading order (lower=better) | 0.042 | 0.044 | PaddleOCR-VL  [arxiv](https://arxiv.org/html/2603.10910v2) |
| Params | 0.9B | 0.9B | Tie |

Both models beat massive frontier systems on this benchmark — GLM-OCR's 94.62 outranks Qwen3-VL-235B (89.15) and Gemini 3 Pro (90.33), and PaddleOCR-VL-1.5 isn't far behind either. [arxiv](https://arxiv.org/html/2603.10910v2)

### Where They Actually Diverge

**Charts and financial/graphical documents:** A direct head-to-head test on 50 pages of Japanese IR (investor relations) materials found PaddleOCR-VL clearly more accurate, especially on graphically dense earnings-report pages, because it has a dedicated, purpose-trained Chart Recognition task — GLM-OCR has no independent chart task and had to be evaluated via its general text pipeline as a substitute, putting it at a disadvantage there. Text-heavy pages (e.g. plain earnings revisions) showed almost no difference between the two. [x](https://x.com/waseda_ai/status/2023834676553805979)

**Speed:** In that same real-world test, PaddleOCR-VL took roughly 2x longer to process than GLM-OCR. GLM-OCR's own benchmarks report throughput of 1.86 pages/second on PDFs, driven by a two-stage architecture (parallel region-layout detection plus multi-token-per-step decoding). [x](https://x.com/waseda_ai/status/2023834676553805979)

**Robustness to real-world scan conditions:** PaddleOCR-VL-1.6 (the newer version) was specifically stress-tested on the Real5-OmniDocBench benchmark — scanning, warping, screen-photography, illumination, and skew — scoring 93.19% overall, with strong results across all five distortion types. GLM-OCR wasn't evaluated on this robustness-focused benchmark in the sources found, so there's no direct comparison there. [arxiv](https://arxiv.org/html/2601.21957v1)

**Version currency:** PaddleOCR-VL has since moved to 1.6, pushing scores to 96.33% on OmniDocBench v1.6 and setting new records on v1.5 and Real5-OmniDocBench too — so if you pull the latest PaddleOCR-VL rather than 1.5, it may now lead GLM-OCR outright on general accuracy, though no direct v1.6-vs-GLM-OCR table was found. [huggingface](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)

## Practical Takeaway 

Tasks specifically figure/table/plot description: PaddleOCR-VL is the better fit for that role given its dedicated chart-recognition training, while GLM-OCR's edge on pure table TEDS scores makes it strong for tabular-data-heavy documents without much charting. If  document corpus leans toward graphs/plots (dashboards, financial charts), prefer PaddleOCR-VL-1.6 for that slot; if it's mostly dense tabular data, GLM-OCR's table scores (93.96/96.39 TEDS) give it a slight lift. Both remain ~0.9B parameters, so either choice keeps low VRAM budget essentially unchanged. [arxiv](https://arxiv.org/html/2603.10910v2)