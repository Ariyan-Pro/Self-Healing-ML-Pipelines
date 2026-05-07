#!/usr/bin/env python3
"""
Main Entry Point for Hugging Face Spaces Deployment
Self-Healing ML Pipelines with Human Veto Dashboard

This file serves as the entry point when deployed to Hugging Face Spaces.
It starts the API server which serves:
- Landing page at /
- Dashboard at /dashboard
- API endpoints at /api/v1/human-veto
"""

import time
from api.api_server import MainAPIServer


def main():
    """Main entry point for Hugging Face Spaces."""
    
    print("="*70)
    print("🚀 Self-Healing ML Pipelines - Hugging Face Spaces")
    print("="*70)
    
    # Create and start API server
    # HF Spaces automatically exposes ports, we use 8080
    server = MainAPIServer(host='0.0.0.0', port=8080)
    
    # Start the server (blocking mode for HF Spaces)
    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.stop()
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
