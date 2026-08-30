# 🚀 FATIGUE CONSOLE // Crew Drift Monitor

### NASA HRP-Inspired Fatigue Model · IBM Granite Decision Support

**Fatigue Console is an AI-assisted mission decision-support system that transforms crew fatigue signals into mission-risk insights, tests intervention tradeoffs across crew and mission tasks, and uses IBM Granite to explain decision options for human review.**

> **Core idea:** Detecting fatigue is only half the problem. An intervention that helps one astronaut now can transfer risk to another crew member, a dependent task, or a later mission day.

---

## 🌌 Selected Challenge Theme

### August's Theme: Advance Space Exploration with AI

Space missions operate in high-stakes environments where human-performance data, schedules, mission tasks, and operational dependencies must be interpreted under limited time and resources.

**Fatigue Console addresses this challenge as an AI-powered space operations and decision-support system.**

Rather than presenting mission personnel with disconnected fatigue measurements, the system connects human-performance signals to mission operations and uses AI to turn those results into understandable decision support.

Our goal is to help move mission operations from:

**Data → Risk → Mission Consequences → Intervention Tradeoffs → Human Decision**

This directly supports the challenge goals of improving mission safety, enabling smarter decisions, and transforming complex space data into actionable insights.

---

## 🎯 Problem Statement

Astronaut fatigue can affect attention, reaction time, and operational performance during demanding missions. However, fatigue risk is not represented by a single measurement.

Relevant information may be distributed across:

- Psychomotor vigilance / reaction-time performance
- Sleep and accumulated sleep debt
- Circadian disruption
- Operational workload

Looking at these signals independently can make developing risk difficult to interpret.

But identifying fatigue is only the first challenge.

Suppose an astronaut assigned to a critical EVA begins showing elevated fatigue risk. Reassigning the task might reduce that astronaut's immediate risk—but could overload the receiving astronaut, affect dependent mission tasks, or increase risk later in the mission.

That creates a broader mission question:

> **How can we combine fatigue-related signals into an interpretable mission-risk picture and help mission personnel understand the downstream consequences of an intervention before making a decision?**

---

## 💡 Solution Description

**Fatigue Console** is a working proof-of-concept that connects crew fatigue monitoring with mission planning and AI-assisted decision support.

The system contains five connected capabilities:

### 1. Fatigue Drift Score

For each astronaut and mission day, the system combines four human-performance signals:

- PVT / reaction-time performance
- Sleep debt
- Circadian phase shift
- Workload

These are transformed by a deterministic model into an interpretable **Fatigue Drift Score** and risk state:

**Nominal → Elevated → High → Critical**

The individual component scores remain visible so users can understand what is driving the result.

### 2. Mission Risk Projection

Individual fatigue states are placed into the wider mission timeline so users can identify when crew risk develops and where the highest projected fatigue risk occurs.

This moves the system beyond a single-person alert toward a mission-level view.

### 3. Historical Replay

Mission personnel can inspect previous mission days to understand how an astronaut reached the current state rather than seeing only the latest score.

### 4. What-If Simulator + Dependency Graph

Users can test hypothetical interventions such as reassigning a demanding task.

The system then evaluates:

- Fatigue score before and after the intervention
- Change in risk level
- Receiving astronaut workload
- Reassignment feasibility
- Later mission trajectory
- Downstream dependent tasks

This exposes an important operational tradeoff:

> **An intervention can reduce local risk while increasing system-level risk elsewhere.**

### 5. IBM Granite Mission Decision Brief

IBM Granite receives the **already-computed, structured mission facts** and translates them into a Mission Decision Brief containing:

- Current astronaut state
- Primary fatigue drivers
- Mission implications
- Intervention assessment
- Receiver workload/fatigue tradeoffs
- Options for consideration
- Uncertainties
- Human-review requirement

Granite does **not** determine the risk score or autonomously make the mission decision.

---

## 🧠 AI Approach and Architecture

Fatigue Console deliberately separates **deterministic risk calculation** from **generative AI explanation**.

In a high-stakes environment, we did not want an LLM inventing risk scores, changing thresholds, or independently deciding whether an astronaut should perform a task.

Instead:

> **The system calculates the facts. IBM Granite explains the facts. Humans make the decision.**

### Architecture

```text
              HUMAN-PERFORMANCE SIGNALS
      ┌──────────┬─────────┬───────────┬──────────┐
      │   PVT    │  Sleep  │ Circadian │ Workload │
      └────┬─────┴────┬────┴─────┬─────┴────┬─────┘
           └───────────┴──────────┴──────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ FATIGUE DRIFT MODEL  │
                │ Deterministic Score  │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │    MISSION RISK      │
                │     PROJECTION       │
                └──────────┬───────────┘
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
   ┌───────────────────┐        ┌───────────────────┐
   │ TASK DEPENDENCIES │        │ WHAT-IF SIMULATOR │
   │ + CASCADE IMPACT  │        │ + FEASIBILITY     │
   └─────────┬─────────┘        └─────────┬─────────┘
             └─────────────┬──────────────┘
                           │
                           ▼
                   VERIFIED CONTEXT
                           │
                           ▼
                  ┌─────────────────┐
                  │   IBM GRANITE   │
                  │ Decision-Support│
                  │ Explanation     │
                  └────────┬────────┘
                           │
                           ▼
               ┌──────────────────────┐
               │ MISSION DECISION     │
               │ BRIEF                │
               │                      │
               │ • What is happening │
               │ • Why it matters    │
               │ • Tradeoffs         │
               │ • Options           │
               │ • Uncertainty       │
               └──────────┬───────────┘
                          │
                          ▼
                    HUMAN REVIEW
