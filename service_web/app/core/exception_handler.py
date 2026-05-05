from fastapi import Request, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from app.core.error_handler import error_response
import logging
import traceback
from fastapi.exceptions import RequestValidationError
from fastapi import status

logger = logging.getLogger(__name__)


async def http_exception_handler(request: Request, exc: HTTPException):
    return error_response(exc.status_code, exc.detail)


def starlette_exception_handler(request, exc):
    if isinstance(exc, RequestValidationError):
        return error_response(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            exc.errors()
        )

    return error_response(
        exc.status_code,
        str(exc.detail)
    )


async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    logger.error(traceback.format_exc())

    error_msg = str(exc)

    return error_response(500, f"An unexpected error occurred: {error_msg}")


async def database_exception_handler(request: Request, exc: SQLAlchemyError):
    logger.error(f"Database error: {str(exc)}")
    logger.error(traceback.format_exc())


    return error_response(500, f"A database error occurred: {str(exc)}")