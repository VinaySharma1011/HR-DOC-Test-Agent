import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import logging
import json
from pathlib import Path

from config import Config

# ════════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT, OUTPUT FORMAT, FALLBACK RESPONSE
# ════════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a professional HR Document Processor Agent. "
    "Your job is to answer HR-related questions, summarize HR policies, and extract relevant information from HR documents. "
    "Always provide clear, concise, and accurate responses based on the provided document context. "
    "If the answer cannot be found in the provided documents, respond with the fallback message."
)
OUTPUT_FORMAT = "Plain text answer. Do not include any code or markdown formatting."
FALLBACK_RESPONSE = "I'm sorry, I could not find the answer to your question in the provided HR documents."

# No SELECTED_DOCUMENT_TITLES section provided, so search all documents.

# ════════════════════════════════════════════════════════════════════════════════
# VALIDATION CONFIG PATH
# ════════════════════════════════════════════════════════════════════════════════

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# ════════════════════════════════════════════════════════════════════════════════
# OBSERVABILITY LIFESPAN FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

app = FastAPI(
    title="HR Document Processor Agent",
    description="Professional agent for answering HR-related questions and extracting information from HR documents.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    lifespan=_obs_lifespan
)

# ════════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING FOR MALFORMED JSON
# ════════════════════════════════════════════════════════════════════════════════

@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request body.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted.",
                "Check for missing commas, colons, or quotes.",
                "Field values must match the expected types.",
                "If sending large text, keep it under 50,000 characters."
            ]
        }
    )

@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    return JSONResponse(
        status_code=400,
        content={
            "success": False,
            "error": "Malformed JSON in request body.",
            "details": str(exc),
            "tips": [
                "Ensure your JSON is properly formatted.",
                "Check for missing commas, colons, or quotes.",
                "If sending large text, keep it under 50,000 characters."
            ]
        }
    )

# ════════════════════════════════════════════════════════════════════════════════
# INPUT VALIDATION UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def _load_validation_config(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _validate_input_model(model: BaseModel, validation_config: Dict[str, str]) -> Optional[str]:
    for field, json_type in validation_config.items():
        value = getattr(model, field, None)
        if json_type == "string":
            if value is not None:
                if not isinstance(value, str):
                    return f"Field '{field}' must be a string."
                if not value.strip():
                    return f"Field '{field}' cannot be empty."
                if len(value) > 50000:
                    return f"Field '{field}' exceeds 50,000 character limit."
        elif json_type == "list":
            if value is not None and not isinstance(value, list):
                return f"Field '{field}' must be a list."
        elif json_type == "int":
            if value is not None and not isinstance(value, int):
                return f"Field '{field}' must be an integer."
        elif json_type == "float":
            if value is not None and not isinstance(value, float):
                return f"Field '{field}' must be a float."
        elif json_type == "bool":
            if value is not None and not isinstance(value, bool):
                return f"Field '{field}' must be a boolean."
        elif json_type == "dict":
            if value is not None and not isinstance(value, dict):
                return f"Field '{field}' must be a dictionary."
    return None

# ════════════════════════════════════════════════════════════════════════════════
# SANITIZE LLM OUTPUT UTILITY
# ════════════════════════════════════════════════════════════════════════════════

import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ════════════════════════════════════════════════════════════════════════════════
# AZURE AI SEARCH CLIENT
# ════════════════════════════════════════════════════════════════════════════════

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
import openai

class AzureAISearchClient:
    """Handles retrieval of relevant document chunks from Azure AI Search."""

    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is None:
            endpoint = Config.AZURE_SEARCH_ENDPOINT
            index_name = Config.AZURE_SEARCH_INDEX_NAME
            api_key = Config.AZURE_SEARCH_API_KEY
            if not endpoint or not index_name or not api_key:
                raise ValueError("Azure AI Search credentials are not configured.")
            self._client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key),
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def retrieve_chunks(self, query: str, k: int = 5) -> list:
        """Retrieve top-k relevant chunks using vector + keyword search."""
        openai_client = openai.AsyncAzureOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
        )
        _t0 = _time.time()
        embedding_resp = await openai_client.embeddings.create(
            input=query,
            model=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002"
        )
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "text-embedding-ada-002",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary="Embedding for retrieval",
            )
        except Exception:
            pass

        vector_query = VectorizedQuery(
            vector=embedding_resp.data[0].embedding,
            k_nearest_neighbors=k,
            fields="vector"
        )
        search_kwargs = {
            "search_text": query,
            "vector_queries": [vector_query],
            "top": k,
            "select": ["chunk", "title"],
        }
        _t1 = _time.time()
        results = self.get_client().search(**search_kwargs)
        try:
            trace_tool_call(
                tool_name="search_client.search",
                latency_ms=int((_time.time() - _t1) * 1000),
                output=str(results)[:200] if results is not None else None,
                status="success",
            )
        except Exception:
            pass
        context_chunks = [r["chunk"] for r in results if r.get("chunk")]
        return context_chunks

# ════════════════════════════════════════════════════════════════════════════════
# CHUNK RETRIEVER
# ════════════════════════════════════════════════════════════════════════════════

class ChunkRetriever:
    """Orchestrates chunk retrieval from Azure AI Search."""

    def __init__(self):
        self.search_client = AzureAISearchClient()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def get_context_chunks(self, query: str, k: int = 5) -> list:
        return await self.search_client.retrieve_chunks(query, k=k)

# ════════════════════════════════════════════════════════════════════════════════
# LLM SERVICE
# ════════════════════════════════════════════════════════════════════════════════

class LLMService:
    """Handles LLM calls to generate answers from context."""

    def __init__(self):
        self._client = None

    def get_client(self):
        if self._client is None:
            api_key = Config.AZURE_OPENAI_API_KEY
            if not api_key:
                raise ValueError("AZURE_OPENAI_API_KEY not configured")
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            )
        return self._client

    async def generate_answer(self, system_prompt: str, context_chunks: list, output_format: str) -> str:
        """Call LLM with system prompt and context, return answer."""
        context_text = "\n\n".join(context_chunks) if context_chunks else ""
        messages = [
            {"role": "system", "content": system_prompt + "\n\nOutput Format: " + output_format},
            {"role": "user", "content": context_text}
        ]
        _t0 = _time.time()
        _llm_kwargs = Config.get_llm_kwargs()
        response = await self.get_client().chat.completions.create(
            model=Config.LLM_MODEL or "gpt-4o",
            messages=messages,
            **_llm_kwargs
        )
        content = response.choices[0].message.content
        try:
            trace_model_call(
                provider="azure",
                model_name=Config.LLM_MODEL or "gpt-4o",
                prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                latency_ms=int((_time.time() - _t0) * 1000),
                response_summary=content[:200] if content else "",
            )
        except Exception:
            pass
        return content

# ════════════════════════════════════════════════════════════════════════════════
# AGENT CLASS
# ════════════════════════════════════════════════════════════════════════════════

class HRDocumentProcessorAgent:
    """Main agent class for HR document processing."""

    def __init__(self):
        self.chunk_retriever = ChunkRetriever()
        self.llm_service = LLMService()

    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process(self) -> Dict[str, Any]:
        """
        Main entry point: retrieves context and generates answer.
        Returns:
            dict: {"success": bool, "result": str, ...}
        """
        validation_config = _load_validation_config(VALIDATION_CONFIG_PATH)
        # No dynamic user input, so no input model to validate

        async with trace_step(
            "retrieve_context",
            step_type="tool_call",
            decision_summary="Retrieve relevant HR document chunks from Azure AI Search",
            output_fn=lambda r: f"{len(r)} chunks",
        ) as step:
            context_chunks = await self.chunk_retriever.get_context_chunks(SYSTEM_PROMPT, k=5)
            step.capture(context_chunks)

        if not context_chunks:
            return {
                "success": False,
                "result": None,
                "error": "No relevant HR document content found.",
                "fallback": FALLBACK_RESPONSE
            }

        async with trace_step(
            "generate_answer",
            step_type="llm_call",
            decision_summary="Generate answer from context using LLM",
            output_fn=lambda r: f"LLM output: {str(r)[:80]}",
        ) as step:
            raw_llm_response = await self.llm_service.generate_answer(
                system_prompt=SYSTEM_PROMPT,
                context_chunks=context_chunks,
                output_format=OUTPUT_FORMAT
            )
            step.capture(raw_llm_response)

        answer = sanitize_llm_output(raw_llm_response, content_type="text")
        if not answer or answer.strip() == "" or answer.strip() == FALLBACK_RESPONSE:
            return {
                "success": False,
                "result": None,
                "error": "No answer found in HR documents.",
                "fallback": FALLBACK_RESPONSE
            }
        return {
            "success": True,
            "result": answer
        }

# ════════════════════════════════════════════════════════════════════════════════
# FASTAPI ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/query", response_model=Dict[str, Any])
@with_content_safety(config=GUARDRAILS_CONFIG)
async def query_endpoint():
    """
    Endpoint to process HR document queries.
    No user input required; agent uses SYSTEM_PROMPT internally.
    """
    agent = HRDocumentProcessorAgent()
    try:
        result = await agent.process()
        return result
    except Exception as e:
        logging.getLogger(__name__).error("Agent processing failed: %s", e, exc_info=True)
        return {
            "success": False,
            "error": f"Agent processing failed: {str(e)}",
            "tips": [
                "Check that Azure AI Search and Azure OpenAI credentials are configured.",
                "Ensure the HR documents are indexed and available.",
                "Contact support if the issue persists."
            ]
        }

# ════════════════════════════════════════════════════════════════════════════════
# AGENT ENTRYPOINT
# ════════════════════════════════════════════════════════════════════════════════

async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())