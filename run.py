"""
Run script for the Feature Detector Testbed.

Usage:
    python run.py
    
Or with custom host/port:
    python run.py --host 0.0.0.0 --port 8080
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Feature Detector Testbed Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"""
╔════════════════════════════════════════════════════════╗
║     Feature Detector Testbed                           ║
╠════════════════════════════════════════════════════════╣
║  Server starting at: http://{args.host}:{args.port}         ║
║  API Docs at:        http://{args.host}:{args.port}/docs    ║
╚════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
