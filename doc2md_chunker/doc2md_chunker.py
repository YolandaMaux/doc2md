"""
Unified entrypoint for the doc2md + chunker service.

Run:
    python doc2md_chunker.py

This will start the FastAPI app defined in doc2md.py, which already:
- Exposes /convert, /image-analysis, /doc2mdchunker
- Imports chunker.ChunkResponse, ChunkerName, chunktext so chunker-style
  chunking is available via /doc2mdchunker and chunker’s own endpoints.
"""
import os
import uvicorn
from doc2md import app  # FastAPI app defined in doc2md.py


if __name__ == "__main__":
    port = int(os.getenv("DOC2MD_EXPOSED_PORT", "8090"))  # default 8080 inside container
    root = os.getenv("DOC2MD_ROOT_PATH", "")  # set ROOT_PATH="" to disable
#    uvicorn.run(app, host="0.0.0.0", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port, root_path=root)
