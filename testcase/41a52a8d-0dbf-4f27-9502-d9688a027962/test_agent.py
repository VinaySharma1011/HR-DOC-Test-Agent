
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from agent import HRDocumentProcessorAgent, FALLBACK_RESPONSE

@pytest.mark.asyncio
async def test_integration_agent_process_returns_fallback_when_no_context():
    """
    Integration test: HRDocumentProcessorAgent.process returns fallback when no context chunks are retrieved.
    Mocks ChunkRetriever.get_context_chunks to return empty list.
    """
    with patch("agent.ChunkRetriever.get_context_chunks", new_callable=AsyncMock) as mock_get_chunks:
        mock_get_chunks.return_value = []
        agent = HRDocumentProcessorAgent()
        result = pytest.run(asyncio_coroutine=agent.process())
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["result"] is None
        assert result["fallback"] == FALLBACK_RESPONSE

@pytest.mark.asyncio
async def test_integration_agent_process_returns_fallback_when_chunkretriever_raises_exception():
    """
    Integration test: HRDocumentProcessorAgent.process returns fallback when ChunkRetriever raises exception.
    Mocks ChunkRetriever.get_context_chunks to raise Exception.
    """
    with patch("agent.ChunkRetriever.get_context_chunks", new_callable=AsyncMock) as mock_get_chunks:
        mock_get_chunks.side_effect = Exception("Search failure")
        agent = HRDocumentProcessorAgent()
        result = pytest.run(asyncio_coroutine=agent.process())
        assert isinstance(result, dict)
        assert result["success"] is False
        assert result["result"] is None
        assert result["fallback"] == FALLBACK_RESPONSE