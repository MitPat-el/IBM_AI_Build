# Astronaut Fatigue & Mission-Risk Decision Support — Backend

A deterministic, interpretable backend API for calculating **Fatigue Drift Scores** from astronaut physiological and operational risk signals.

> ⚠️ **Safety Notice:** This system is a mission-planning decision-support tool. It is **not** a medical device or diagnosis system. All final operational decisions must remain with qualified human personnel.

---

## Design Principles

| Principle | Implementation |
|---|---|
| **AI never calculates the score** | `engine/scoring.py` uses pure math only — no LLM calls |
| **AI only explains** | `engine/ai_explain.py` reads an existing score; IBM watsonx generates narrative |
| **Single source of truth** | All weights and thresholds live only in `app/config.py` |
| **Human in the loop** | Every API response includes a disclaimer; no auto-action is triggered |
| **Modular** | Each concern (scoring, AI, routes) is its own file for easy extension |

---

## Project Structure

```
fatigue-backend/
├── app/
│   ├── main.py               ← FastAPI app + router registration
│   ├── config.py             ← Signal weights & risk-level thresholds (edit here only)
│   ├── models/
│   │   └── fatigue.py        ← Pydantic request/response schemas
│   ├── engine/
│   │   ├── scoring.py        ← Deterministic scoring engine (no AI)
│   │   └── ai_explain.py     ← IBM watsonx explanation layer (stub → real)
│   └── routes/
│       ├── fatigue.py        ← POST /fatigue/calculate
│       └── ai_explain.py     ← POST /ai/explain
├── tests/
│   ├── test_scoring.py       ← Unit tests for the scoring engine
│   └── test_api.py           ← Integration tests for the HTTP routes
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
cd fatigue-backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the server

```bash
uvicorn app.main:app --reload
```

The API is now live at **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

---

## API Reference

### `POST /fatigue/calculate`

Calculates a Fatigue Drift Score deterministically from raw risk signals.

**Request body:**
```json
{
  "astronaut_id": "CDR-001",
  "timestamp": "2025-01-15T14:30:00Z",
  "pvt_risk": 72.0,
  "sleep_risk": 65.0,
  "circadian_risk": 80.0,
  "workload_risk": 55.0
}
```

All risk values are `0–100`. Missing or out-of-range values return `422 Unprocessable Entity`.

**Example response:**
```json
{
  "astronaut_id": "CDR-001",
  "timestamp": "2025-01-15T14:30:00Z",
  "fatigue_score": 68.1,
  "risk_level": "MODERATE",
  "signal_breakdown": [
    { "signal": "pvt_risk",       "raw_value": 72.0, "weight": 0.30, "weighted_contribution": 21.6 },
    { "signal": "sleep_risk",     "raw_value": 65.0, "weight": 0.30, "weighted_contribution": 19.5 },
    { "signal": "circadian_risk", "raw_value": 80.0, "weight": 0.20, "weighted_contribution": 16.0 },
    { "signal": "workload_risk",  "raw_value": 55.0, "weight": 0.20, "weighted_contribution": 11.0 }
  ],
  "top_contributing_factors": ["pvt_risk", "sleep_risk", "circadian_risk", "workload_risk"],
  "disclaimer": "This score is a decision-support tool for mission planning only. ..."
}
```

**Quick curl test:**
```bash
curl -X POST http://localhost:8000/fatigue/calculate \
  -H "Content-Type: application/json" \
  -d '{
    "astronaut_id": "CDR-001",
    "timestamp": "2025-01-15T14:30:00Z",
    "pvt_risk": 72,
    "sleep_risk": 65,
    "circadian_risk": 80,
    "workload_risk": 55
  }'
```

---

### `POST /ai/explain`

Submits a complete `FatigueResult` to IBM watsonx.ai and returns a natural-language explanation.

> The AI **reads** the score — it does not recalculate it.

**Request body:** the full JSON from `/fatigue/calculate`

**Activation:** Set the following environment variables:
```bash
export WATSONX_API_KEY="your-api-key"
export WATSONX_PROJECT_ID="your-project-id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"   # default
```

Until credentials are set, the endpoint returns a stub message showing the prompt that _would_ be sent to watsonx.

---

## Fatigue Drift Score Formula

```
Fatigue Drift Score =
  (pvt_risk       × 0.30) +
  (sleep_risk     × 0.30) +
  (circadian_risk × 0.20) +
  (workload_risk  × 0.20)
```

### Risk Levels

| Score | Level |
|---|---|
| 0 – 39 | 🟢 LOW |
| 40 – 69 | 🟡 MODERATE |
| 70 – 84 | 🟠 HIGH |
| 85 – 100 | 🔴 CRITICAL |

> ⚙️ **These weights and thresholds are prototype assumptions.** Edit **only** `app/config.py` to update them when your research team provides validated values. No other file needs to change.

---

## Running Tests

```bash
pytest tests/ -v
```

Expected output: all tests pass with no warnings.

---

## Adding Future Modules

The architecture is designed for extension. To add a new module:

1. **Mission Risk Projection** → create `app/engine/mission_risk.py` + `app/routes/mission_risk.py`
2. **What-If Simulator** → create `app/engine/whatif.py` + `app/routes/whatif.py`
3. **Historical Replay** → add `app/db/` layer + `app/routes/history.py`
4. **New risk signal** → add its weight to `config.SIGNAL_WEIGHTS` (keep sum = 1.0), add the field to `FatigueInput`, and add the key to the `signals` dict in `scoring.calculate_fatigue_score()`

In all cases: register the new router in `app/main.py` with `app.include_router(...)`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `WATSONX_API_KEY` | No (for stub) | IBM watsonx API key |
| `WATSONX_PROJECT_ID` | No (for stub) | IBM watsonx project ID |
| `WATSONX_URL` | No | watsonx endpoint (default: us-south) |
