import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.router import router
from app.api.structure import router as structure_router
from app.application.planner_service import seed
from app.hardware.repository import HardwareRepository
from app.hardware.router import router as hardware_router
from app.infrastructure.database import ROOT, database, migrate
from app.integrations.huggingface import SourceUnavailable
from app.repositories.catalog import CatalogRepository, MissingResource


def create_app(database_url: str | None = None, static_dir: Path | None = None, importer=None,
               *, enable_workload=True):
    url = database_url or os.environ.get("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'planner.db'}")
    if not database_url and "DATABASE_URL" not in os.environ:
        (ROOT / "data").mkdir(exist_ok=True)
    engine, sessions = database(url)

    @asynccontextmanager
    async def lifespan(app):
        migrate(engine)
        with sessions() as session:
            seed(CatalogRepository(session))
            HardwareRepository(session).seed()
        worker = None
        if enable_workload:
            from app.workload.jobs import Worker
            worker = Worker(sessions)
            app.state.workload_worker = worker
            worker.start()
        yield
        if worker:
            worker.stop()
        engine.dispose()

    app = FastAPI(title="AI Workload Planner", version="0.1.0", lifespan=lifespan)
    app.state.engine, app.state.sessions, app.state.importer = engine, sessions, importer
    if enable_workload:
        from app.workload.router import router as workload_router
        app.include_router(workload_router)
    app.include_router(router)
    app.include_router(structure_router)
    app.include_router(hardware_router)

    @app.exception_handler(MissingResource)
    async def missing(_request: Request, exc: MissingResource):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(SourceUnavailable)
    async def unavailable(_request: Request, exc: SourceUnavailable):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(_request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    dist = static_dir or ROOT / "frontend" / "dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path == "api" or path.startswith(("api/", "assets/")):
            raise HTTPException(404, "Not found")
        if "." in path or ".." in path or path.startswith("."):
            raise HTTPException(404, "Not found")
        index = dist / "index.html"
        if not index.is_file():
            raise HTTPException(503, "Frontend not built. Run npm ci && npm run build in frontend/")
        return FileResponse(index, headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
