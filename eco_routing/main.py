from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from eco_routing.api.routes import router as eco_router
from eco_routing.api.health import router as health_router
from eco_routing.api.auth import router as auth_router
from eco_routing.core.road_graph import build_global_graph


def create_app() -> FastAPI:
    app = FastAPI(title="GreenRoute AI - Eco Routing")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )
    app.include_router(health_router, prefix="/eco-route")
    app.include_router(auth_router, prefix="/auth")
    app.include_router(eco_router, prefix="/eco-route")

    @app.on_event("startup")
    def startup_event():
        # build_global_graph()  # Commenting out for faster startup
        print("App started successfully - graph building skipped for now")

    return app


app = create_app()