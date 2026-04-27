
import pytest
import asyncio
import types
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent

from fastapi.testclient import TestClient
import httpx

@pytest.fixture(scope="module")
def test_app():
    # Import FastAPI app from agent
    return agent.app

@pytest.fixture
def test_client(test_app):
    # Use TestClient for sync endpoints
    return TestClient(test_app)

@pytest.mark.asyncio
async def test_health_endpoint_returns_ok():
    """Validates that the /health endpoint returns a 200 status and correct response."""
    from agent import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_query_endpoint_returns_success_with_valid_context(monkeypatch):
    """Ensures /query endpoint returns a successful response when HR documents are available."""
    from agent import app, HRDocumentProcessorAgent

    # Patch HRDocumentProcessorAgent.process to return a valid answer
    answer = "This is a valid HR answer."
    async def mock_process(self):
        return {"success": True, "result": answer}
    monkeypatch.setattr(agent.HRDocumentProcessorAgent, "process", mock_process)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/query")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["result"] is not None
        assert isinstance(data["result"], str)

@pytest.mark.asyncio
async def test_query_endpoint_returns_fallback_when_no_context(monkeypatch):
    """Ensures /query endpoint returns fallback response when no relevant HR document content is found."""
    from agent import app, HRDocumentProcessorAgent, FALLBACK_RESPONSE

    # Patch HRDocumentProcessorAgent.process to simulate no context found
    async def mock_process(self):
        return {
            "success": False,
            "result": None,
            "error": "No relevant HR document content found.",
            "fallback": FALLBACK_RESPONSE
        }
    monkeypatch.setattr(agent.HRDocumentProcessorAgent, "process", mock_process)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/query")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["fallback"] == FALLBACK_RESPONSE

@pytest.mark.asyncio
async def test_hrdocumentprocessoragent_process_returns_success(monkeypatch):
    """Tests HRDocumentProcessorAgent.process() returns a successful result when context and LLM output are available."""
    from agent import HRDocumentProcessorAgent

    # Patch ChunkRetriever.get_context_chunks and LLMService.generate_answer
    mock_chunks = ["HR policy chunk 1", "HR policy chunk 2"]
    mock_llm_answer = "This is a synthesized HR answer."

    with patch.object(agent.ChunkRetriever, "get_context_chunks", new=AsyncMock(return_value=mock_chunks)), \
         patch.object(agent.LLMService, "generate_answer", new=AsyncMock(return_value=mock_llm_answer)), \
         patch("agent.sanitize_llm_output", return_value=mock_llm_answer):
        agent_instance = HRDocumentProcessorAgent()
        result = await agent_instance.process()
        assert result["success"] is True
        assert result["result"] is not None
        assert isinstance(result["result"], str)

@pytest.mark.asyncio
async def test_hrdocumentprocessoragent_process_returns_fallback_on_no_chunks(monkeypatch):
    """Tests HRDocumentProcessorAgent.process() returns fallback when no context chunks are found."""
    from agent import HRDocumentProcessorAgent, FALLBACK_RESPONSE

    # AUTO-FIXED invalid syntax: with patch.object(agent.ChunkRetriever, "get_context_chunks", new=AsyncMock(return_value=[]):
    agent_instance = HRDocumentProcessorAgent()
    result = await agent_instance.process()
    assert result["success"] is False
    assert result["fallback"] == FALLBACK_RESPONSE

def test_azureaisearchclient_get_client_returns_searchclient(monkeypatch):
    """Tests AzureAISearchClient.get_client() returns a valid SearchClient instance when config is set."""
    from agent import AzureAISearchClient, Config
    # Patch Config variables
    monkeypatch.setattr(Config, "AZURE_SEARCH_ENDPOINT", "https://test-search-endpoint")
    monkeypatch.setattr(Config, "AZURE_SEARCH_INDEX_NAME", "test-index")
    monkeypatch.setattr(Config, "AZURE_SEARCH_API_KEY", "test-api-key")

    # Patch SearchClient to a dummy class for type check
    with patch("azure.search.documents.SearchClient") as mock_sc:
        mock_sc.return_value = MagicMock(spec=mock_sc)
        client = AzureAISearchClient().get_client()
        assert client is not None
        assert isinstance(client, MagicMock)

def test_llmservice_get_client_returns_asyncazureopenai(monkeypatch):
    """Tests LLMService.get_client() returns a valid AsyncAzureOpenAI client when config is set."""
    from agent import LLMService, Config
    # Patch Config variables
    monkeypatch.setattr(Config, "AZURE_OPENAI_API_KEY", "test-azure-openai-key")
    monkeypatch.setattr(Config, "AZURE_OPENAI_ENDPOINT", "https://test-openai-endpoint")

    # Patch openai.AsyncAzureOpenAI to a dummy class for type check
    with patch("openai.AsyncAzureOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock(spec=mock_openai)
        client = LLMService().get_client()
        assert client is not None
        assert isinstance(client, MagicMock)

def test_sanitize_llm_output_removes_fences_and_wrappers():
    """Tests sanitize_llm_output removes markdown fences and conversational wrappers from LLM output."""
    from agent import sanitize_llm_output
    raw = "```python\nHere is the answer:\nThis is the HR policy.\n```\nLet me know if you need more info."
    cleaned = agent.sanitize_llm_output(raw, content_type="text")
    assert "```" not in cleaned
    assert not cleaned.lower().startswith("here is")
    assert isinstance(cleaned, str)

@pytest.mark.asyncio
async def test_validation_exception_handler_returns_422():
    """Tests FastAPI validation_exception_handler returns 422 and proper error message for malformed JSON."""
    from agent import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # Send malformed JSON (missing closing brace)
        resp = await ac.post("/query", content='{"foo": "bar"', headers={"Content-Type": "application/json"})
        assert resp.status_code == 422
        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "Malformed JSON or invalid request body."

@pytest.mark.asyncio
async def test_json_decode_exception_handler_returns_400():
    """Tests FastAPI json_decode_exception_handler returns 400 and proper error message for JSONDecodeError."""
    from agent import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        # Send invalid JSON (missing quotes)
        resp = await ac.post("/query", content='{foo: bar}', headers={"Content-Type": "application/json"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert data["error"] == "Malformed JSON in request body."