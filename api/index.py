import sys
import os

# Ensure backend directory is in sys.path for Vercel Serverless Function execution
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
backend_dir = os.path.join(root_dir, "backend")

if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from app.main import app
    handler = app
except Exception as _import_err:
    import traceback
    _tb = traceback.format_exc()

    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    _error_app = FastAPI()

    @_error_app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def _startup_error_handler(path: str):
        return JSONResponse(
            {
                "error": str(_import_err),
                "type": type(_import_err).__name__,
                "traceback": _tb,
                "startup": "FAILED"
            },
            status_code=500
        )

    handler = _error_app
