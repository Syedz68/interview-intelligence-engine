from fastapi.responses import JSONResponse
from typing import Union, Dict, Any

def error_response(status_code: int, message: Union[str, Dict[str, Any]]):
    if isinstance(message, list):
        # RequestValidationError passes a list of error dicts
        msg = "; ".join(
            [f"{' -> '.join(str(loc) for loc in err.get('loc', []))}: {err.get('msg', '')}" for err in message]
        )
    elif isinstance(message, dict):
        msg = message.get("message", str(message))
    else:
        msg = message

    return JSONResponse(
        status_code=status_code,
        content={
            "status_code": status_code,
            "message": msg,
            "data": None
        }
    )