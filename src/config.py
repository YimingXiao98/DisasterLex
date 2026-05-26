"""
Centralized configuration for the ContextGraphRAG pipeline.

All settings are read from environment variables (loaded from .env via
load_dotenv) with sensible defaults. Import the singleton `cfg` instead of
calling os.getenv() in individual agent files.

Usage
-----
    from src.config import cfg

    llm = ChatOpenAI(model=cfg.LLM_MODEL, base_url=cfg.LLM_BASE_URL, ...)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _bool(env_var: str, default: bool = True) -> bool:
    """Parse a boolean env var. Accepts '1', 'true', 'yes', 'on' as True."""
    val = os.getenv(env_var)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    # ── API Keys ──────────────────────────────────────────────────────────────
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    NVIDIA_API_KEY: str = field(
        default_factory=lambda: os.getenv("NVIDIA_API_KEY", "")
    )
    DASHSCOPE_API_KEY: str = field(
        default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", "")
    )
    DEEPSEEK_API_KEY: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", "")
    )
    DEEPSEEK_BASE_URL: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    VLLM_BASE_URL: str = field(
        default_factory=lambda: os.getenv("VLLM_BASE_URL", "")
    )
    VLLM_API_KEY: str = field(
        default_factory=lambda: os.getenv("VLLM_API_KEY", "EMPTY")
    )
    DEEPSEEK_THINKING: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_THINKING", "disabled")
    )
    DEEPSEEK_REASONING_EFFORT: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_REASONING_EFFORT", "high")
    )
    DASHSCOPE_BASE_URL: str = field(
        default_factory=lambda: os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    )
    TAVILY_API_KEY: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    # ── LLM Settings ─────────────────────────────────────────────────────────
    LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "google/gemini-2.5-flash")
    )
    LLM_BASE_URL: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    )
    RECURSION_LIMIT: int = field(
        default_factory=lambda: int(os.getenv("RECURSION_LIMIT", "50"))
    )
    OPENROUTER_RESPONSE_CACHE: bool = field(
        default_factory=lambda: _bool("OPENROUTER_RESPONSE_CACHE", default=False)
    )
    OPENROUTER_RESPONSE_CACHE_TTL: int = field(
        default_factory=lambda: int(os.getenv("OPENROUTER_RESPONSE_CACHE_TTL", "3600"))
    )

    # ── Tool Feature Flags ────────────────────────────────────────────────────
    ENABLE_WEB_SEARCH: bool = field(
        default_factory=lambda: _bool("ENABLE_WEB_SEARCH", default=True)
    )
    ENABLE_KNOWLEDGE_GRAPH: bool = field(
        default_factory=lambda: _bool("ENABLE_KNOWLEDGE_GRAPH", default=True)
    )
    ENABLE_DATABASE: bool = field(
        default_factory=lambda: _bool("ENABLE_DATABASE", default=True)
    )

    # ── Tavily Settings ───────────────────────────────────────────────────────
    TAVILY_MAX_RESULTS: int = field(
        default_factory=lambda: int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    )
    TAVILY_TOPIC: str = field(
        default_factory=lambda: os.getenv("TAVILY_TOPIC", "general")
    )

    # ── Neo4j Settings ────────────────────────────────────────────────────────
    NEO4J_URI: str = field(
        default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687")
    )
    NEO4J_USER: str = field(
        default_factory=lambda: os.getenv("NEO4J_USER", "neo4j")
    )
    NEO4J_PASSWORD: str = field(
        default_factory=lambda: os.getenv("NEO4J_PASSWORD", "password")
    )

    def validate(self) -> None:
        """Raise ValueError for required keys that are missing."""
        if self.ENABLE_WEB_SEARCH and not self.TAVILY_API_KEY:
            raise ValueError(
                "TAVILY_API_KEY not found but ENABLE_WEB_SEARCH=true. "
                "Set TAVILY_API_KEY in your .env file or set ENABLE_WEB_SEARCH=false."
            )


NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Model-name prefixes that should route to NVIDIA's dedicated API
# (instead of the default OpenRouter endpoint).
_NVIDIA_PREFIXES: tuple[str, ...] = (
    "nvidia/",
    "deepseek-ai/",
)

# Model-name prefixes that should route to Alibaba Cloud DashScope's
# OpenAI-compatible endpoint. DashScope native model IDs use names like
# "qwen-plus", "qwen-max", "qwen3-max", "qwen3.6-flash", "qwen-turbo" with no provider/slash -
# distinct from OpenRouter's "qwen/qwen3.5-..." slugs.
_DASHSCOPE_PREFIXES: tuple[str, ...] = (
    "qwen-",
    "qwen3-",
    "qwen3.",
    "qwen2-",
    "qwen2.5-",
    "qwq-",
    "qvq-",
)

_DEEPSEEK_NATIVE_MODELS: tuple[str, ...] = (
    "deepseek-v4-flash",
    "deepseek-v4-pro",
    "deepseek-chat",
    "deepseek-reasoner",
)

# Model names with this prefix route to a self-hosted vLLM OpenAI-compat
# endpoint pointed at by VLLM_BASE_URL. The portion after "local/" is sent
# to vLLM as the model id, so it must match vLLM's --served-model-name.
_VLLM_PREFIX = "local/"


def resolve_provider(model_name: str) -> tuple[str, str]:
    """Return (base_url, api_key) for the given model name.

    Prefix-based auto-routing:
      - models with ``local/*`` prefix -> self-hosted vLLM (VLLM_BASE_URL)
      - DeepSeek native model IDs (``deepseek-v4-flash``, ``deepseek-v4-pro``)
        -> official DeepSeek OpenAI-compatible API
      - models with NVIDIA-style prefixes (``nvidia/*``, ``deepseek-ai/*``)
        -> NVIDIA integrate API
      - models with DashScope-native prefixes (``qwen-*``, ``qwen3-*``, ``qwq-*``,
        ``qvq-*``, etc.; no slash) -> Alibaba Cloud DashScope (OpenAI-compatible)
      - everything else -> OpenRouter (default)

    This lets ``--pipeline-model qwen-plus`` transparently hit DashScope and
    ``--pipeline-model moonshotai/kimi-k2-instruct`` hit NVIDIA without any
    other config change. Use the ``qwen/...`` slug form to keep OpenRouter.
    """
    name = (model_name or "").lower()
    if name.startswith(_VLLM_PREFIX):
        if not cfg.VLLM_BASE_URL:
            raise ValueError(
                f"Model '{model_name}' has local/ prefix but VLLM_BASE_URL is not set. "
                "Start a vLLM server (e.g. scripts/baselines/vllm_server.sbatch) "
                "and export VLLM_BASE_URL=http://<gpu-node>:<port>/v1 before running."
            )
        return cfg.VLLM_BASE_URL, cfg.VLLM_API_KEY
    if name in _DEEPSEEK_NATIVE_MODELS:
        if not cfg.DEEPSEEK_API_KEY:
            raise ValueError(
                f"Model '{model_name}' requires DEEPSEEK_API_KEY but it's not set in .env."
            )
        return cfg.DEEPSEEK_BASE_URL, cfg.DEEPSEEK_API_KEY
    if any(name.startswith(p) for p in _NVIDIA_PREFIXES):
        if not cfg.NVIDIA_API_KEY:
            raise ValueError(
                f"Model '{model_name}' requires NVIDIA_API_KEY but it's not set in .env."
            )
        return NVIDIA_BASE_URL, cfg.NVIDIA_API_KEY
    if any(name.startswith(p) for p in _DASHSCOPE_PREFIXES):
        if not cfg.DASHSCOPE_API_KEY:
            raise ValueError(
                f"Model '{model_name}' requires DASHSCOPE_API_KEY but it's not set in .env."
            )
        return cfg.DASHSCOPE_BASE_URL, cfg.DASHSCOPE_API_KEY
    if not cfg.OPENROUTER_API_KEY:
        raise ValueError(
            f"Model '{model_name}' requires OPENROUTER_API_KEY but it's not set in .env."
        )
    return cfg.LLM_BASE_URL, cfg.OPENROUTER_API_KEY


def provider_request_kwargs(model_name: str, base_url: str) -> dict[str, object]:
    """Return provider-specific OpenAI-compatible request kwargs.

    DeepSeek V4 defaults to thinking mode. The current LangChain tool loop does
    not preserve DeepSeek's reasoning_content across tool-call turns, so direct
    DeepSeek runs default to non-thinking mode unless DEEPSEEK_THINKING=enabled
    is set explicitly. The same DEEPSEEK_THINKING toggle also disables reasoning
    for deepseek-v4 slugs routed via OpenRouter, which otherwise add ~5x
    per-call latency in our benchmark pipeline.
    """
    name = (model_name or "").lower()
    url = (base_url or "").lower()
    thinking = cfg.DEEPSEEK_THINKING.strip().lower()
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("DEEPSEEK_THINKING must be 'enabled' or 'disabled'.")

    if name in _DEEPSEEK_NATIVE_MODELS and "api.deepseek.com" in url:
        kwargs: dict[str, object] = {"extra_body": {"thinking": {"type": thinking}}}
        if thinking == "enabled":
            effort = cfg.DEEPSEEK_REASONING_EFFORT.strip().lower()
            if effort not in {"high", "max"}:
                raise ValueError("DEEPSEEK_REASONING_EFFORT must be 'high' or 'max'.")
            kwargs["reasoning_effort"] = effort
        return kwargs

    # Models on OpenRouter that default to reasoning/thinking mode — our
    # benchmark pipeline doesn't need their chain-of-thought (it has its own
    # ReAct loops), and enabling it adds 3-10× per-call latency.
    if "openrouter.ai" in url and any(tag in name for tag in ("deepseek-v4", "hy3-preview", "hunyuan-", "qwen3.6-")):
        return {"extra_body": {"reasoning": {"enabled": thinking == "enabled"}}}

    return {}


def provider_default_headers(base_url: str, title: str = "Disaster Context Graph Agent") -> dict[str, str]:
    """Return provider-specific default headers for OpenAI-compatible clients."""
    headers = {
        "HTTP-Referer": "http://localhost",
        "X-Title": title,
    }
    if "openrouter.ai" in (base_url or "").lower() and cfg.OPENROUTER_RESPONSE_CACHE:
        headers["X-OpenRouter-Cache"] = "true"
        headers["X-OpenRouter-Cache-TTL"] = str(cfg.OPENROUTER_RESPONSE_CACHE_TTL)
    return headers


# ── Module-level singleton ───────────────────────────────────────────────────
cfg: Config = Config()
