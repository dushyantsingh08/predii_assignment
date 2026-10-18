"""Central configuration.

Everything tunable lives here and can be overridden with environment variables
(or a ``.env`` file), so the same code runs on a laptop with no API key
(offline mode) or against OpenAI / Anthropic / a local Ollama model.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------
# Manual-specific layout knowledge.
# These describe how *this family* of PDFs (browser-printed Ford workshop
# manual pages) is laid out.  Adapting to another manual mostly means editing
# these few constants.
# --------------------------------------------------------------------------

# Lines that are print artefacts rather than manual content.
NOISE_LINE_PATTERNS = (
    r"^Page \d+ (?:sur|of) \d+$",          # "Page 1 sur 2" (browser print header)
    r"^\d{4}\s+.+\s+Workshop Manual$",       # "2014 F-150 Workshop Manual"
    r"^file:///",                            # print footer with the source path
    r"^about:blank",
    r"^repair4less$",                        # watermark
    r"^\d{4}-\d{2}-\d{2}$",                  # print date
)

TOP_NOISE_Y = 80.0        # running page header lives above this (pt)
BOTTOM_NOISE_Y = 760.0    # running page footer lives below this (pt)
HEADER_BLOCK_Y = 150.0    # "SECTION xxx / doc type / revision" block (page 1 of a doc)
HEADER_BLOCK_MAX_FONT = 8.6


@dataclass
class Settings:
    # ---- input / cache -----------------------------------------------------
    pdf_path: str = field(default_factory=lambda: _env("SPECX_PDF", "data/service-manual.pdf"))
    cache_dir: str = field(default_factory=lambda: _env("SPECX_CACHE", ".specx_cache"))

    # ---- PDF parsing -------------------------------------------------------
    ocr_fallback: bool = True          # OCR pages that have no text layer (needs tesseract)
    ocr_min_chars: int = 25
    para_gap_pt: float = 14.0          # vertical gap that starts a new paragraph
    row_tolerance_pt: float = 3.5      # lines within this y-distance form one visual row

    # ---- chunking ----------------------------------------------------------
    max_chunk_words: int = 260
    min_chunk_words: int = 45
    table_max_rows: int = 18           # split long spec tables, repeating the header

    # ---- embeddings --------------------------------------------------------
    # sentence-transformers | openai | tfidf (offline fallback, no downloads)
    embed_backend: str = field(default_factory=lambda: _env("SPECX_EMBED_BACKEND", "sentence-transformers"))
    embed_model: str = field(default_factory=lambda: _env("SPECX_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    openai_embed_model: str = field(default_factory=lambda: _env("SPECX_OPENAI_EMBED_MODEL", "text-embedding-3-small"))

    # ---- retrieval ---------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("SPECX_TOP_K", 6))
    candidate_k: int = 40              # candidates pulled from each retriever before fusion
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    intent_boost: float = 1.30         # multiplicative boost for chunk types matching intent
    spec_doc_boost: float = 1.10       # SPECIFICATIONS documents are authoritative
    coverage_weight: float = 0.6       # lexical rerank: x(1-w/2 .. 1+w/2) by share of query terms present

    # ---- LLM ---------------------------------------------------------------
    # auto | anthropic | openai | none   ("none" => deterministic table parser only)
    llm_provider: str = field(default_factory=lambda: _env("SPECX_LLM_PROVIDER", "auto"))
    llm_model: str = field(default_factory=lambda: _env("SPECX_LLM_MODEL", ""))
    llm_base_url: str = field(default_factory=lambda: _env("SPECX_LLM_BASE_URL", ""))
    llm_max_tokens: int = 1800
    max_results: int = 12

    def resolved_llm_provider(self) -> str:
        """Pick a provider from the environment when set to ``auto``."""
        p = self.llm_provider.lower()
        if p != "auto":
            return p
        if _env("ANTHROPIC_API_KEY"):
            return "anthropic"
        if _env("OPENAI_API_KEY") or self.llm_base_url:
            return "openai"
        return "none"

    def resolved_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "anthropic": "claude-sonnet-5-5",
            "openai": "gpt-4o-mini",
        }.get(self.resolved_llm_provider(), "")
