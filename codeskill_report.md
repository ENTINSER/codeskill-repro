# CODESKILL Prototype Experiment Report

Generated at: 2026-07-17T08:50:28.486484 UTC

## 1. Experiment Setup

- LLM backend: `kimi`
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Number of tasks: 30
- Error types: 10
- Task order: grouped (by error type)
- Skill dedup threshold: 0.85
- Deprecation threshold: success_rate < 0.3 and uses >= 3
- Merge similarity threshold: 0.9
- Retrieval: top_k=5, threshold=0.3, recent-skill boost enabled

## 2. Success Rate by Configuration

| Configuration | Successes | Total | Success Rate |
|---|---:|---:|---:|
| no_skill | 17 | 30 | 56.7% |
| static_skill | 18 | 30 | 60.0% |
| codeskill | 18 | 30 | 60.0% |

## 3. CODESKILL Skill Bank Summary

- Total skills created: 18
- Active skills: 17
- Deprecated skills: 1
- Skill extraction success rate: 18/18 (100.0%)
- Final average active success rate: 73.8%

## 4. Error-Type Breakdown (CODESKILL)

| Error Type | Successes | Total |
|---|---:|---:|
| ConcurrencyRace | 3 | 3 |
| ConnectionTimeout | 2 | 3 |
| FileNotFoundError | 1 | 3 |
| IndexOutOfBounds | 0 | 3 |
| InfiniteLoop | 3 | 3 |
| LogicError | 2 | 3 |
| NullPointerException | 1 | 3 |
| ResourceLeak | 0 | 3 |
| TypeError | 3 | 3 |
| ZeroDivisionError | 3 | 3 |

## 5. Key Observations

- CODESKILL extracts reusable skills from successful repairs and retrieves them for future tasks.
- Static skills provide a fixed baseline; CODESKILL evolves the bank based on actual task outcomes.
- Deprecation and merging prevent the skill bank from growing indefinitely and remove low-quality skills.

## 6. Artifacts

- `codeskill_experiment_data.json`: raw experiment logs
- `codeskill_success_rates.png`: success rate bar chart
- `codeskill_skill_count.png`: skill count over tasks
- `codeskill_avg_success.png`: average skill success rate over tasks
