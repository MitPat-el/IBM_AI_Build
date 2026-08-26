"""
main.py — FastAPI application entrypoint.  v0.2.0

Architecture:
  The deterministic model calculates risk.
  IBM AI explains the result.
  Human mission personnel make the decision.

NASA research grounds the variables and simulated ranges used in this prototype.
The current score weights, thresholds, mission-risk formulas, and demonstration
records are prototype assumptions and are NOT NASA-validated operational models.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.db.database import init_db
from app.api import fatigue, mission, what_if, history, ai

DEMO_DIR = Path(__file__).parent / "demo"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the database on startup."""
    init_db()
    yield


app = FastAPI(
    title="Astronaut Fatigue & Mission-Risk Decision Support API",
    description=(
        "**The deterministic model calculates risk. "
        "IBM AI explains the result. "
        "Human mission personnel make the decision.**\n\n"
        "A transparent, interpretable backend for astronaut fatigue scoring, "
        "mission risk projection, what-if intervention simulation, and "
        "IBM watsonx-powered explanations.\n\n"
        "> ⚠️ This system is a mission-planning decision-support prototype. "
        "It is NOT a medical device, NASA-validated operational tool, or "
        "autonomous decision-making system."
    ),
    version="0.2.0",
    contact={"name": "Hackathon Team"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
app.include_router(fatigue.router)
app.include_router(mission.router)
app.include_router(what_if.router)
app.include_router(history.router)
app.include_router(ai.router)

# ---------------------------------------------------------------------------
# Developer demo page (single HTML file, no framework)
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=str(DEMO_DIR)), name="static")


@app.get("/demo", include_in_schema=False)
def demo_page():
    """Serve the developer demo/test page."""
    return FileResponse(str(DEMO_DIR / "index.html"))


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "service": "Astronaut Fatigue & Mission-Risk Decision Support API",
        "version": "0.2.0",
        "docs": "/docs",
        "architecture": (
            "Deterministic engine calculates risk → "
            "IBM AI explains → Human decides"
        ),
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
