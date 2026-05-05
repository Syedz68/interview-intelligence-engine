"""
service_web  –  Web gateway / main API service
===============================================
Runs on port 8000.
Handles: video ingestion, CV pipeline (frames, gaze, motion, emotion,
         landmarks), AI analysis (fluency, Q&A, behavioral report),
         and proxies audio work to service_audio on port 8001.

Environment requirements
------------------------
Uses .venv_web  (numpy≥2, opencv, mediapipe, deepface, fastapi).

Start
-----
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.core.exception_handler import (
    http_exception_handler,
    starlette_exception_handler,
    global_exception_handler,
    database_exception_handler,
)
from app.api.v1.routes.process import router as process_router
from app.api.v2.routes.process import router as new_process_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(StarletteHTTPException, starlette_exception_handler)
app.add_exception_handler(Exception, global_exception_handler)
app.add_exception_handler(SQLAlchemyError, database_exception_handler)

app.include_router(process_router)
app.include_router(new_process_router)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )
    openapi_schema.setdefault("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    openapi_schema["security"] = [{"BearerAuth": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/")
def home():
    return {"message": "Welcome to the Interview Intelligence Core Engine"}
