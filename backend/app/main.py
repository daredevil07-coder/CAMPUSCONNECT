import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.services.seed_service import seed_database
from app.services.rag.hybrid_search import hybrid_search_engine
from app.services.gemini_service import gemini_service, quota_manager
from app.api.v1.router import api_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("campusconnect")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle (lightweight for serverless execution)."""
    logger.info("CampusConnect serverless instance ready.")
    yield
    logger.info("CampusConnect shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="CampusConnect — Official AI Campus Assistant for Lovely Professional University (LPU), Phagwara, Punjab.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Router
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["infrastructure"])
@app.get(f"{settings.API_V1_STR}/health", tags=["infrastructure"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "CampusConnect LPU Assistant",
        "version": settings.VERSION,
        "environment": settings.APP_ENV,
        "indexed_records": len(hybrid_search_engine.corpus_cache),
    }


@app.get(f"{settings.API_V1_STR}/health/ai", tags=["infrastructure"])
async def ai_health_check():
    """Gemini AI provider availability and quota health status."""
    return {
        "gemini_configured": gemini_service.is_configured(),
        "configured_model": settings.GEMINI_MODEL,
        "circuit_open": quota_manager.is_circuit_open(),
        "quota_stats": quota_manager.get_stats(),
        "verified_fallback_available": True,
    }


# Serve built frontend static files if available (after explicit API routes)
import os
from fastapi.staticfiles import StaticFiles

frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
