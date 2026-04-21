"""
SecureWealth Twin AI — FastAPI application entry point.

Middleware is registered in reverse order of execution:
  add_middleware(PIIMaskingMiddleware)      ← runs last
  add_middleware(ConsentValidationMiddleware) ← runs second
  add_middleware(TraceIDMiddleware)         ← runs first (added last)
"""

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.middleware import (
    ConsentValidationMiddleware,
    PIIMaskingMiddleware,
    TraceIDMiddleware,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="SecureWealth Twin AI",
    version="0.1.0",
    description=(
        "AI-powered wealth advisory and cyber-protection backend. "
        "All outputs are for simulation and informational purposes only."
    ),
)

# Register middleware — Starlette applies them in reverse add_middleware order,
# so TraceID runs first, then ConsentValidation, then PIIMasking.
app.add_middleware(PIIMaskingMiddleware)
app.add_middleware(ConsentValidationMiddleware)
app.add_middleware(TraceIDMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "trace_id": trace_id},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    trace_id = getattr(request.state, "trace_id", str(uuid.uuid4()))
    logger.error("Unhandled exception trace_id=%s: %s", trace_id, str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred",
            "trace_id": trace_id,
        },
    )


@app.on_event("startup")
async def startup_event():
    import asyncio

    logger.info("SecureWealth Twin AI starting up...")

    # Eagerly initialize all components so the first request doesn't pay the cost
    loop = asyncio.get_event_loop()
    try:
        from app.api.routes import get_orchestrator

        orchestrator = await loop.run_in_executor(None, get_orchestrator)
        logger.info("All components initialized successfully")

        # Verify vector DB is populated
        if not orchestrator._rag_pipeline.ping():
            logger.warning("Vector DB is empty or not queryable on startup")
        else:
            logger.info("Vector DB is populated and queryable")

        # Attempt market data fetch (non-blocking — failure does not abort startup)
        try:
            from app.engines.market_data import MarketDataClient

            client = MarketDataClient()
            snapshot = client.fetch()
            if snapshot.is_stale:
                logger.warning(
                    "Market data provider unavailable on startup; using stale/null data"
                )
            else:
                logger.info("Market data provider connectivity verified")
        except Exception as e:
            logger.warning("Market data provider check failed on startup: %s", e)

    except Exception as e:
        logger.error("Startup initialization failed: %s", e)
        raise


@app.on_event("startup")
async def startup_event():
    import asyncio

    logger.info("SecureWealth Twin AI starting up...")

    loop = asyncio.get_event_loop()
    try:
        from app.api.routes import get_orchestrator

        orchestrator = await loop.run_in_executor(None, get_orchestrator)
        logger.info("All components initialized successfully")

        if not orchestrator._rag_pipeline.ping():
            logger.warning("Vector DB is empty or not queryable on startup")
        else:
            logger.info("Vector DB is populated and queryable")

        try:
            from app.engines.market_data import MarketDataClient

            client = MarketDataClient()
            snapshot = client.fetch()
            if snapshot.is_stale:
                logger.warning("Market data provider unavailable on startup; using stale/null data")
            else:
                logger.info("Market data provider connectivity verified")
        except Exception as e:
            logger.warning("Market data provider check failed on startup: %s", e)

    except Exception as e:
        logger.error("Startup initialization failed: %s", e)
        raise


from app.api.routes import router  # noqa: E402
app.include_router(router)
