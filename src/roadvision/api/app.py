"""FastAPI application factory and static hosting."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .routes import agent_routes, auth_routes, benchmark_routes, harbor_routes, studio_routes


def create_app() -> FastAPI:
    app = FastAPI(
        title="RoadVision AI — Synthetic Visual Reasoning Benchmark",
        description="Visual Task Factory for Road Pavement Defect Evaluation",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(auth_routes.router)
    app.include_router(agent_routes.router)
    app.include_router(studio_routes.router)
    app.include_router(benchmark_routes.router)
    app.include_router(harbor_routes.router)

    # Healthcheck
    @app.get("/health")
    def healthcheck():
        return {
            "status": "healthy",
            "service": "RoadVision AI",
            "edition": "Enterprise Infrastructure Intelligence",
            "track": "Visual Task Factory",
        }

    # Mount static assets if web UI exists
    web_dir = Path(__file__).parent.parent / "web"
    static_dir = web_dir / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Serve index HTML for frontend
    @app.get("/", response_class=HTMLResponse)
    def index():
        index_file = web_dir / "index.html"
        if index_file.exists():
            return index_file.read_text(encoding="utf-8")
        return "<h1>RoadVision AI API is Running.</h1><p>Visit <a href='/api/docs'>API Documentation</a>.</p>"

    return app


app = create_app()
