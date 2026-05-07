"""
API endpoints for Self-Healing ML Pipelines.

Available endpoints:
    - HumanVetoAPI: Human override and veto endpoint (always available)
    - MainAPIServer: Main API server hosting all endpoints with Web UI
    
The Human Veto endpoint is ALWAYS AVAILABLE as a safety mechanism,
providing both REST API endpoints and a web-based dashboard UI.

REST API Endpoints:
    GET    /api/v1/human-veto          - List pending vetoes
    POST   /api/v1/human-veto          - Submit new veto request
    PUT    /api/v1/human-veto/<id>     - Approve/Reject a veto
    DELETE /api/v1/human-veto/<id>     - Cancel a veto request
    GET    /api/v1/human-veto/history  - Get veto history
    GET    /health                     - Health check endpoint
    
Web UI:
    GET    /                           - Human Veto Dashboard
"""

from .human_veto_endpoint import HumanVetoAPI, VetoRequest, VetoStore
from .api_server import MainAPIServer

__all__ = ['HumanVetoAPI', 'VetoRequest', 'VetoStore', 'MainAPIServer']
