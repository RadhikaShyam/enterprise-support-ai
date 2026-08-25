import logging
import time
import uuid

from fastapi import Request


logger = logging.getLogger("api")


async def request_logging_middleware(
    request: Request,
    call_next,
):

    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    try:

        response = await call_next(request)

        duration = (
            time.perf_counter() - start_time
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        logger.info(
            "request_completed "
            "request_id=%s "
            "method=%s "
            "path=%s "
            "status=%s "
            "duration_ms=%.2f",

            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration * 1000,
        )

        return response

    except Exception:

        duration = (
            time.perf_counter() - start_time
        )

        logger.exception(
            "request_failed "
            "request_id=%s "
            "method=%s "
            "path=%s "
            "duration_ms=%.2f",

            request_id,
            request.method,
            request.url.path,
            duration * 1000,
        )

        raise