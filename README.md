# Rational Negotiator

Hybrid LLM + deterministic rules negotiation agent for the AgentBeats MAizeBargAIn benchmark.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Incoming Message                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   parse_observation()                        │
│              Extract JSON from message text                  │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────────┐
│   ACCEPT_OR_REJECT      │     │         PROPOSE             │
│   (Deterministic)       │     │    (LLM + Constraints)      │
├─────────────────────────┤     ├─────────────────────────────┤
│ value >= BATNA? ACCEPT  │     │ 1. prepare_context()        │
│ value < BATNA?  WALK    │     │ 2. call_llm()               │
│                         │     │ 3. enforce_constraints()    │
│ No LLM needed           │     │    M3 → M2 → M1             │
└─────────────────────────┘     └─────────────────────────────┘
```

## M1-M5 Constraint Rules

| Rule | Description | Priority | Action |
|------|-------------|----------|--------|
| **M1** | Don't decrease allocation_other vs previous offer | Low | Increase to match previous |
| **M2** | My value must be >= BATNA | High | Recalculate or WALK |
| **M3** | Don't offer 0 items or keep 0 items | Medium | Give/take 1 least valuable item |
| **M4** | Don't accept offer worse than BATNA | - | Handled in ACCEPT_OR_REJECT |
| **M5** | Don't WALK from offer better than BATNA | - | Handled in ACCEPT_OR_REJECT |

### Priority Order

```
M5 > M3 > M4 > M2 > M1

In enforce_constraints(): M3 → M2 → M1
```

### Conflict Resolution

- **M1 vs M2**: M2 wins. If M1 forces allocation below BATNA, recalculate via M2.
- **M2 vs M3**: M3 applied first, then M2 checked. If impossible to satisfy both → WALK.

## Supported Providers

| Provider | API Key Variable | Default Model |
|----------|------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o-mini` |
| OpenRouter | `OPENROUTER_API_KEY` | `openai/gpt-4o-mini` |

OpenAI takes priority if both keys are set.

## Quick Start

### 1. Install dependencies

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### 2. Configure environment

```bash
cp sample.env .env
# Edit .env and add your API key
```

### 3. Run the agent

```bash
# Using uv
uv run python main.py --host 0.0.0.0 --port 8080

# Or directly
python main.py --host 0.0.0.0 --port 8080
```

### 4. Run with Docker

```bash
docker build -t rational-negotiator .
docker run -p 8080:8080 --env-file .env rational-negotiator
```

## Testing Against Green Agent

### 1. Start green agent

```bash
cd ../tutorial-agent-beats-comp/scenarios/bargaining
python bargaining_green.py serve --port 9029
```

### 2. Start purple agent

```bash
cd ../rational-negotiator
python main.py --port 8080
```

### 3. Run evaluation

```bash
cd ../tutorial-agent-beats-comp/scenarios/bargaining
python bargaining_green.py once --config test_config.json
```

Example `test_config.json`:
```json
{
  "participants": {
    "challenger": "http://localhost:8080/"
  },
  "challenger_label": "rational",
  "games": 10,
  "max_rounds": 5,
  "full_matrix": false,
  "model": "nfsp",
  "circle": 0
}
```

## Response Formats

### ACCEPT_OR_REJECT

```json
{"action": "ACCEPT"}
{"action": "WALK"}
```

### PROPOSE

```json
{
  "allocation_self": [3, 2, 4],
  "allocation_other": [2, 3, 1]
}
```

Where:
- `allocation_self[i] + allocation_other[i] = quantities[i]`
- `allocation_self` = what I keep
- `allocation_other` = what opponent receives

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | One of these | OpenAI API key |
| `OPENROUTER_API_KEY` | One of these | OpenRouter API key |
| `LLM_MODEL` | No | Override default model |
| `PORT` | No | Server port (default: 8080) |
| `HOST` | No | Server host (default: 0.0.0.0) |

## License

MIT
