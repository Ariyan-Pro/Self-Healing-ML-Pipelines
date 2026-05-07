#!/usr/bin/env python3
"""
Human Veto API Endpoint for Self-Healing ML Pipelines

Provides a REST API endpoint for human operators to veto, override, or approve
autonomous healing actions. This endpoint is always available as a safety mechanism.

Endpoints:
    POST   /api/v1/human-veto     - Submit a veto for a pending action
    GET    /api/v1/human-veto     - List pending actions awaiting human review
    PUT    /api/v1/human-veto/{id} - Approve or reject a specific action
    DELETE /api/v1/human-veto/{id} - Cancel a veto request
    GET    /api/v1/human-veto/history - Get veto history

Usage:
    python human_veto_endpoint.py --port 8080
    
    # Or import as a module:
    from api.human_veto_endpoint import HumanVetoAPI
    api = HumanVetoAPI(port=8080)
    api.start()

Author: Self-Healing ML Pipelines Team
License: MIT
"""

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs
import uuid


class VetoRequest:
    """Represents a human veto request."""
    
    def __init__(
        self,
        action_id: str,
        action_type: str,
        reason: str,
        priority: str = "normal",
        metadata: Optional[Dict] = None
    ):
        self.id = str(uuid.uuid4())
        self.action_id = action_id
        self.action_type = action_type
        self.reason = reason
        self.priority = priority  # low, normal, high, critical
        self.metadata = metadata or {}
        self.status = "pending"  # pending, approved, rejected, cancelled
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.decision_by: Optional[str] = None
        self.decision_at: Optional[str] = None
        self.notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'action_id': self.action_id,
            'action_type': self.action_type,
            'reason': self.reason,
            'priority': self.priority,
            'metadata': self.metadata,
            'status': self.status,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'decision_by': self.decision_by,
            'decision_at': self.decision_at,
            'notes': self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VetoRequest':
        veto = cls(
            action_id=data['action_id'],
            action_type=data['action_type'],
            reason=data['reason'],
            priority=data.get('priority', 'normal'),
            metadata=data.get('metadata')
        )
        veto.id = data.get('id', veto.id)
        veto.status = data.get('status', 'pending')
        veto.created_at = data.get('created_at', veto.created_at)
        veto.updated_at = data.get('updated_at', veto.updated_at)
        veto.decision_by = data.get('decision_by')
        veto.decision_at = data.get('decision_at')
        veto.notes = data.get('notes')
        return veto


class VetoStore:
    """In-memory store for veto requests with file persistence."""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path('logs/veto_store.json')
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._requests: Dict[str, VetoRequest] = {}
        self._lock = threading.Lock()
        self._load()
    
    def _load(self):
        """Load existing veto requests from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    for req_data in data:
                        veto = VetoRequest.from_dict(req_data)
                        self._requests[veto.id] = veto
                print(f"📂 Loaded {len(self._requests)} veto requests from storage")
            except Exception as e:
                print(f"⚠️  Could not load veto store: {e}")
    
    def _save(self):
        """Persist veto requests to disk."""
        try:
            with open(self.storage_path, 'w') as f:
                data = [req.to_dict() for req in self._requests.values()]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save veto store: {e}")
    
    def add(self, veto: VetoRequest) -> VetoRequest:
        """Add a new veto request."""
        with self._lock:
            self._requests[veto.id] = veto
            self._save()
        return veto
    
    def get(self, veto_id: str) -> Optional[VetoRequest]:
        """Get a veto request by ID."""
        with self._lock:
            return self._requests.get(veto_id)
    
    def get_all(self, status_filter: Optional[str] = None) -> List[VetoRequest]:
        """Get all veto requests, optionally filtered by status."""
        with self._lock:
            requests = list(self._requests.values())
            if status_filter:
                requests = [r for r in requests if r.status == status_filter]
            # Sort by priority then creation time
            priority_order = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
            requests.sort(key=lambda r: (priority_order.get(r.priority, 2), r.created_at))
            return requests
    
    def update(self, veto_id: str, updates: Dict) -> Optional[VetoRequest]:
        """Update a veto request."""
        with self._lock:
            veto = self._requests.get(veto_id)
            if veto:
                for key, value in updates.items():
                    if hasattr(veto, key):
                        setattr(veto, key, value)
                veto.updated_at = datetime.now().isoformat()
                self._save()
            return veto
    
    def delete(self, veto_id: str) -> bool:
        """Delete a veto request."""
        with self._lock:
            if veto_id in self._requests:
                del self._requests[veto_id]
                self._save()
                return True
            return False
    
    def get_history(self, limit: int = 100) -> List[VetoRequest]:
        """Get historical (non-pending) veto requests."""
        with self._lock:
            history = [r for r in self._requests.values() if r.status != 'pending']
            history.sort(key=lambda r: r.decision_at or r.updated_at, reverse=True)
            return history[:limit]


class HumanVetoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Human Veto API."""
    
    store: VetoStore = None  # Will be set by server
    
    def log_message(self, format, *args):
        """Custom logging format."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {args[0]}")
    
    def send_json_response(self, data: Any, status_code: int = 200):
        """Send a JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def send_error_response(self, message: str, status_code: int = 400):
        """Send an error response."""
        self.send_json_response({'error': message}, status_code)
    
    def read_request_body(self) -> Optional[Dict]:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return None
        
        try:
            body = self.rfile.read(content_length)
            return json.loads(body.decode())
        except Exception as e:
            return None
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        
        if path == '/' or path == '/index.html':
            # Serve the Human Veto Dashboard UI
            self.serve_dashboard()
        
        elif path == '/api/v1/human-veto' or path == '/api/v1/human-veto/':
            # List pending vetoes
            status_filter = query.get('status', [None])[0]
            vetoes = self.store.get_all(status_filter)
            self.send_json_response({
                'count': len(vetoes),
                'veto_requests': [v.to_dict() for v in vetoes]
            })
        
        elif path == '/api/v1/human-veto/history':
            # Get veto history
            limit = int(query.get('limit', [100])[0])
            history = self.store.get_history(limit)
            self.send_json_response({
                'count': len(history),
                'veto_requests': [v.to_dict() for v in history]
            })
        
        elif path.startswith('/api/v1/human-veto/'):
            # Get specific veto
            veto_id = path.split('/')[-1]
            veto = self.store.get(veto_id)
            if veto:
                self.send_json_response(veto.to_dict())
            else:
                self.send_error_response(f'Veto request {veto_id} not found', 404)
        
        elif path == '/health' or path == '/health/':
            # Health check endpoint
            self.send_json_response({
                'status': 'healthy',
                'service': 'human-veto-api',
                'timestamp': datetime.now().isoformat(),
                'pending_vetoes': len(self.store.get_all('pending'))
            })
        
        else:
            self.send_error_response('Not found', 404)
    
    def serve_dashboard(self):
        """Serve the Human Veto Dashboard HTML page."""
        dashboard_path = Path(__file__).parent / 'veto_dashboard.html'
        if dashboard_path.exists():
            try:
                with open(dashboard_path, 'r') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content.encode())
            except Exception as e:
                self.send_error_response(f'Error loading dashboard: {e}', 500)
        else:
            self.send_json_response({
                'message': 'Human Veto API is running',
                'endpoints': {
                    'list': 'GET /api/v1/human-veto',
                    'create': 'POST /api/v1/human-veto',
                    'update': 'PUT /api/v1/human-veto/<id>',
                    'delete': 'DELETE /api/v1/human-veto/<id>',
                    'history': 'GET /api/v1/human-veto/history',
                    'health': 'GET /health'
                }
            })
    
    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path == '/api/v1/human-veto' or path == '/api/v1/human-veto/':
            # Create new veto request
            body = self.read_request_body()
            if not body:
                self.send_error_response('Invalid or missing request body')
                return
            
            required_fields = ['action_id', 'action_type', 'reason']
            for field in required_fields:
                if field not in body:
                    self.send_error_response(f'Missing required field: {field}')
                    return
            
            veto = VetoRequest(
                action_id=body['action_id'],
                action_type=body['action_type'],
                reason=body['reason'],
                priority=body.get('priority', 'normal'),
                metadata=body.get('metadata')
            )
            
            self.store.add(veto)
            self.send_json_response(veto.to_dict(), 201)
        
        else:
            self.send_error_response('Not found', 404)
    
    def do_PUT(self):
        """Handle PUT requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith('/api/v1/human-veto/'):
            parts = path.split('/')
            if len(parts) < 5:
                self.send_error_response('Invalid URL format')
                return
            
            veto_id = parts[-1]
            veto = self.store.get(veto_id)
            
            if not veto:
                self.send_error_response(f'Veto request {veto_id} not found', 404)
                return
            
            body = self.read_request_body()
            if not body:
                self.send_error_response('Invalid or missing request body')
                return
            
            # Validate decision
            if 'status' in body:
                if body['status'] not in ['approved', 'rejected']:
                    self.send_error_response('Status must be "approved" or "rejected"')
                    return
                
                updates = {
                    'status': body['status'],
                    'decision_by': body.get('decision_by', 'anonymous'),
                    'decision_at': datetime.now().isoformat(),
                    'notes': body.get('notes')
                }
                
                veto = self.store.update(veto_id, updates)
                self.send_json_response(veto.to_dict())
            else:
                self.send_error_response('Missing "status" field in request body')
        
        else:
            self.send_error_response('Not found', 404)
    
    def do_DELETE(self):
        """Handle DELETE requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        if path.startswith('/api/v1/human-veto/'):
            veto_id = path.split('/')[-1]
            
            if self.store.delete(veto_id):
                self.send_json_response({'message': f'Veto request {veto_id} deleted'})
            else:
                self.send_error_response(f'Veto request {veto_id} not found', 404)
        
        else:
            self.send_error_response('Not found', 404)


class HumanVetoAPI:
    """Main API server class."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080, storage_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.store = VetoStore(storage_path)
        self.server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
    
    def start(self, blocking: bool = True):
        """Start the API server."""
        HumanVetoHandler.store = self.store
        
        self.server = HTTPServer((self.host, self.port), HumanVetoHandler)
        
        print(f"\n{'='*60}")
        print(f"🛡️  Human Veto API Server Starting")
        print(f"{'='*60}")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Storage: {self.store.storage_path}")
        print(f"\n📡 Endpoints:")
        print(f"   GET    http://{self.host}:{self.port}/api/v1/human-veto")
        print(f"   POST   http://{self.host}:{self.port}/api/v1/human-veto")
        print(f"   PUT    http://{self.host}:{self.port}/api/v1/human-veto/<id>")
        print(f"   DELETE http://{self.host}:{self.port}/api/v1/human-veto/<id>")
        print(f"   GET    http://{self.host}:{self.port}/api/v1/human-veto/history")
        print(f"   GET    http://{self.host}:{self.port}/health")
        print(f"\n✅ Server is ALWAYS AVAILABLE for human override")
        print(f"{'='*60}\n")
        
        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                print("\n👋 Shutting down server...")
                self.stop()
        else:
            self._thread = threading.Thread(target=self.server.serve_forever)
            self._thread.daemon = True
            self._thread.start()
    
    def stop(self):
        """Stop the API server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            print("✅ Server stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Human Veto API Endpoint for Self-Healing ML Pipelines',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python human_veto_endpoint.py --port 8080
    python human_veto_endpoint.py --host 127.0.0.1 --port 9000
    python human_veto_endpoint.py --storage /path/to/veto_store.json

API Usage Examples:
    # List pending vetoes
    curl http://localhost:8080/api/v1/human-veto
    
    # Submit a veto
    curl -X POST http://localhost:8080/api/v1/human-veto \\
         -H "Content-Type: application/json" \\
         -d '{"action_id": "heal-123", "action_type": "retrain", "reason": "Unexpected data pattern"}'
    
    # Approve a veto
    curl -X PUT http://localhost:8080/api/v1/human-veto/<id> \\
         -H "Content-Type: application/json" \\
         -d '{"status": "approved", "decision_by": "operator@example.com"}'
    
    # Health check
    curl http://localhost:8080/health
        """
    )
    
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Port to listen on')
    parser.add_argument('--storage', '-s', type=str, default=None, help='Path to veto storage file')
    parser.add_argument('--test', action='store_true', help='Run in test mode (exit after startup)')
    
    args = parser.parse_args()
    
    api = HumanVetoAPI(host=args.host, port=args.port, storage_path=args.storage)
    
    if args.test:
        print("🧪 Test mode - verifying server can start...")
        api.start(blocking=False)
        time.sleep(2)
        api.stop()
        print("✅ Test passed")
        return 0
    else:
        api.start(blocking=True)
        return 0


if __name__ == '__main__':
    sys.exit(main())
