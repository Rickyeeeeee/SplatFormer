"""FastAPI application for the SplatFormer evaluation viewer."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .index import EvaluationDataError, EvaluationIndex


def create_app(eval_dir: Path, reference_iteration: str = "00000000", web_dist: Optional[Path] = None) -> FastAPI:
    index = EvaluationIndex(eval_dir, reference_iteration=reference_iteration)
    app = FastAPI(title="SplatFormer Evaluation Viewer", version="1.0.0")
    app.state.evaluation_index = index

    @app.get("/api/health")
    def health():
        return {"status": "ok", **index.status()}

    @app.get("/api/iterations")
    def iterations():
        return index.iterations()

    @app.post("/api/refresh")
    def refresh():
        reference_panel.cache_clear()
        return index.refresh()

    @app.get("/api/iterations/{iteration}/scenes")
    def scenes(
        iteration: str,
        sort: str = Query("gain_psnr"),
        order: str = Query("desc"),
        search: str = Query(""),
        page: int = Query(1, ge=1),
        page_size: int = Query(200, ge=1, le=500),
    ):
        try:
            return index.scenes(iteration, sort, order, search, page, page_size)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/iterations/{iteration}/scenes/{scene_name}/views")
    def views(
        iteration: str,
        scene_name: str,
        sort: str = Query("gain_psnr"),
        order: str = Query("desc"),
        search: str = Query(""),
        page: int = Query(1, ge=1),
        page_size: int = Query(500, ge=1, le=500),
    ):
        try:
            return index.views(iteration, scene_name, sort, order, search, page, page_size)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, EvaluationDataError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @lru_cache(maxsize=512)
    def reference_panel(scene_name: str, image_name: str, kind: str) -> bytes:
        return index.reference_image(scene_name, image_name, kind)

    @app.get("/api/images/{iteration}/{scene_name}/{image_name}/{kind}")
    def image(iteration: str, scene_name: str, image_name: str, kind: str):
        headers = {"Cache-Control": "public, max-age=3600"}
        try:
            if kind == "prediction":
                return FileResponse(index.prediction_path(iteration, scene_name, image_name), headers=headers)
            if kind in {"input", "gt"}:
                return Response(reference_panel(scene_name, image_name, kind), media_type="image/png", headers=headers)
            raise HTTPException(status_code=404, detail=f"Unknown image kind: {kind}")
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EvaluationDataError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    dist = Path(web_dist).resolve() if web_dist else Path(__file__).parent / "web" / "dist"
    if dist.is_dir() and (dist / "index.html").is_file():
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

        @app.get("/{spa_path:path}", include_in_schema=False)
        def spa(spa_path: str):
            candidate = (dist / spa_path).resolve()
            if dist in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        def missing_frontend():
            return {
                "message": "Backend is running. Build eval_viewer/web or use its Vite development server.",
                "api_docs": "/docs",
            }

    return app
