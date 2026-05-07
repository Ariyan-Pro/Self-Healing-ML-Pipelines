#!/usr/bin/env python3
"""
Main API Server for Self-Healing ML Pipelines

Integrates all API endpoints including the Human Veto endpoint.
This server is ALWAYS AVAILABLE as a safety mechanism for human operators.

Usage:
    python api_server.py --port 8080
    
Author: Self-Healing ML Pipelines Team
License: MIT
"""

import argparse
import sys
import threading
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from http.server import HTTPServer
from api.human_veto_endpoint import HumanVetoAPI, VetoStore


class MainAPIServer:
    """
    Main API Server that hosts all endpoints for the Self-Healing ML Pipeline.
    
    This server ensures the Human Veto endpoint is ALWAYS AVAILABLE,
    even when the main pipeline is not running.
    """
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080, storage_path: str = None):
        self.host = host
        self.port = port
        self.storage_path = storage_path
        self.human_veto_api: HumanVetoAPI = None
        self.server: HTTPServer = None
        self._running = False
        
    def start(self, blocking: bool = True):
        """Start the main API server with all endpoints."""
        
        # Initialize the Human Veto API (ALWAYS AVAILABLE)
        self.human_veto_api = HumanVetoAPI(
            host=self.host,
            port=self.port,
            storage_path=self.storage_path
        )
        
        print(f"\n{'='*70}")
        print(f"🚀 Self-Healing ML Pipeline API Server")
        print(f"{'='*70}")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"\n📡 Available Endpoints:")
        print(f"   ─────────────────────────────────────────────────────")
        print(f"   HUMAN VETO ENDPOINT (Always Available):")
        print(f"   GET    /api/v1/human-veto          - List pending vetoes")
        print(f"   POST   /api/v1/human-veto          - Submit new veto")
        print(f"   PUT    /api/v1/human-veto/<id>     - Approve/Reject veto")
        print(f"   DELETE /api/v1/human-veto/<id>     - Cancel veto request")
        print(f"   GET    /api/v1/human-veto/history  - Get veto history")
        print(f"   ─────────────────────────────────────────────────────")
        print(f"   HEALTH CHECK:")
        print(f"   GET    /health                     - Service health status")
        print(f"   ─────────────────────────────────────────────────────")
        print(f"   WEB UI:")
        print(f"   GET    /                           - Human Veto Dashboard")
        print(f"{'='*70}")
        print(f"✅ Human Veto Endpoint is ALWAYS AVAILABLE for override")
        print(f"{'='*70}\n")
        
        # Start the server
        self.human_veto_api.start(blocking=blocking)
        self._running = True
        
    def stop(self):
        """Stop the API server."""
        if self.human_veto_api:
            self.human_veto_api.stop()
        self._running = False
        print("✅ API Server stopped")
        
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


def main():
    parser = argparse.ArgumentParser(
        description='Self-Healing ML Pipeline API Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python api_server.py --port 8080
    python api_server.py --host 127.0.0.1 --port 9000
    python api_server.py --storage /path/to/veto_store.json

The Human Veto endpoint will be available at:
    http://localhost:8080/api/v1/human-veto

Web Dashboard available at:
    http://localhost:8080/
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Port to listen on (default: 8080)'
    )
    
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Path to veto store JSON file'
    )
    
    args = parser.parse_args()
    
    server = MainAPIServer(
        host=args.host,
        port=args.port,
        storage_path=args.storage
    )
    
    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        print("\n👋 Received shutdown signal...")
        server.stop()
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
