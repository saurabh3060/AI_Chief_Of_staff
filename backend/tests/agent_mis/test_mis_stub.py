"""
tests/agent_mis/test_mis_stub.py
===================================
Verifies the MIS Insight agent stub: the route is reachable and returns
a clean HTTP 501 (not a 404 or unhandled crash), and the underlying
service/graph functions raise `AgentNotImplementedError` as documented.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_get_metric_insights_raises_not_implemented():
    from core.exceptions import AgentNotImplementedError
    from agent_mis.service import get_metric_insights

    with pytest.raises(AgentNotImplementedError):
        await get_metric_insights()


@pytest.mark.asyncio
async def test_mis_graph_raises_not_implemented_when_invoked():
    from core.exceptions import AgentNotImplementedError
    from agent_mis.graph import mis_graph

    with pytest.raises(AgentNotImplementedError):
        await mis_graph.ainvoke({"insights": []})


def test_get_insights_route_returns_501(required_env):
    from main import app

    with TestClient(app) as client:
        response = client.get("/agents/mis/insights")

    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()
