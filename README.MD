# Chat Over Tabular Data

RAG-style question-answering over Excel tables (Clients, Invoices, InvoiceLineItems) using Text-to-SQL and LLMs.

## Design principles

- **Reliable, API-based system with fallback:** The primary model is **OpenAI** (e.g. `gpt-4o-mini`). On API failure or rate limit, the system falls back to **Anthropic** (e.g. `claude-sonnet-4-20250514`). This applies to both SQL generation and answer generation. See [src/config.py](src/config.py).
- **Two pipeline options:** (1) **General pipeline** ([src/pipeline.py](src/pipeline.py)): linear flow (generate SQL → validate → execute → generate answer). (2) **LangGraph-based pipeline** ([src/pipeline_langgraph.py](src/pipeline_langgraph.py)): agent with tools (get_schema, validate_sql, execute_sql) and an explicit answer node; supports ReAct and retries within the graph. Use the LangGraph pipeline for an agent-style loop (e.g. `python tools/run_eval.py --langgraph` or `run_pipeline_langgraph(question)` in code). The Streamlit app uses the normal (linear) pipeline by default.

## High-level system structure

The system has two main parts:

1. **SQL generator (LLM):** User natural-language question → LLM generates a SQL query. Implemented in [src/text_to_sql.py](src/text_to_sql.py) (`generate_sql`), with schema and prompts built in [src/schema.py](src/schema.py).
2. **Answer generator:** Retrieved table data (SQL result) → LLM produces a natural-language answer. Implemented in [src/llm_client.py](src/llm_client.py) (`generate_answer`).

## Data and eval files

- **Data (Excel):** All table data must be in **`data/`**: `data/Clients.xlsx`, `data/Invoices.xlsx`, `data/InvoiceLineItems.xlsx`
- **Evaluation:** Benchmark, test set, and few-shot examples live in **`evaluation/`**:
  - **`evaluation/benchmark.json`** – evaluation questions with `expected_sql` and `expected_answer` (used for metrics; built from data via `tools/build_benchmark.py`)
  - **`evaluation/test_set.json`** – held-out test questions (15 items); used for test-set eval or CSV export
  - **`evaluation/few_shot_examples.json`** – few-shot (question, sql) examples in simple terms, used in prompts and retrieval
  - **`evaluation/eval_results.json`** – written when you run the evaluation script (includes model names and answer similarity metrics)
  - **`evaluation/test_set_results.csv`** – optional: question, sql, agent_final_response for the test set (see Evaluation below)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY)
```

## Run

```bash
streamlit run app.py
```

## Problems and solutions – SQL generation

**Problem:** The SQL agent must produce valid, schema-consistent SQL from a user query.

- **Domain knowledge:** [src/domain_knowledge.py](src/domain_knowledge.py) defines table semantics, column semantics, country regions, allowed join paths, time semantics, and revenue/billing rules (e.g. when to include tax). This is injected into the schema prompt in [src/schema.py](src/schema.py) so the model understands the business context.
- **In-context learning (few-shot):** Few-shot (question, SQL) examples are loaded from [evaluation/few_shot_examples.json](evaluation/few_shot_examples.json) (or [src/retrieval.py](src/retrieval.py) for retrieval-based selection) and added to the prompt so the model sees table relationships and example queries.
- **Vector database (retrieval):** [src/retrieval.py](src/retrieval.py) uses **Chroma** as a vector store and **OpenAI embeddings** (text-embedding-3-small) to retrieve relevant schema chunks and few-shot examples by semantic similarity when retrieval is enabled. Schema and few-shot examples are embedded and stored in in-memory Chroma collections; the question is embedded and used to fetch the most relevant tables and example (question, SQL) pairs. If Chroma is not installed or the embedding API is unavailable, the pipeline falls back to the full schema and file-based few-shot from `evaluation/few_shot_examples.json`.
- **SQL checker:** [src/sql_validator.py](src/sql_validator.py) validates on three criteria: **syntax** (sqlparse), **safety** (only SELECT/WITH; forbidden keywords such as DROP, DELETE, INSERT, etc.), and **schema** (EXPLAIN QUERY PLAN against the DB). If any check fails, validation returns an error message.
- **Retry mechanism:** The pipeline retries SQL generation up to `MAX_SQL_RETRIES` (e.g. 3) when validation fails; the last validation error is fed back to the LLM so it can fix the query. If all retries fail, the system does not execute invalid SQL; it returns a clear error to the user. See [src/pipeline.py](src/pipeline.py) and [src/config.py](src/config.py).

## Problems and solutions – Answer generation

**Problem:** The answer generator must produce a valid answer from the retrieved table and must not invent information when the SQL result already contains the answer or when information is insufficient.

**Solution:** Prompt design in [src/llm_client.py](src/llm_client.py) (`generate_answer`): (1) The model is told that the SQL was written to answer the question and that filtering was already applied. (2) It must base the answer strictly on the provided rows and not infer extra data. (3) It must decide whether the question can be fully answered from the result; if not, it should state what is missing or ask a clarifying question. (4) It must not claim "insufficient data" merely because some columns are omitted—only when the question logically needs information not present in the rows. This addresses the case when the exact answer is already in the SQL result versus when the answer generator has more decision-making responsibility.

## Code structure

| Area | File | Role |
|------|------|------|
| Config | [src/config.py](src/config.py) | Models (OpenAI + Anthropic fallback), `MAX_ROWS_FOR_ANSWER`, `MAX_SQL_RETRIES` |
| Data | [src/data_loader.py](src/data_loader.py) | Load Excel tables, build in-memory SQLite DB |
| Schema and prompts | [src/schema.py](src/schema.py) | Build full prompt (structure, relations, semantics, usage constraints, few-shot, CoT/ReAct) |
| Domain knowledge | [src/domain_knowledge.py](src/domain_knowledge.py) | Table/column semantics, country regions, revenue/tax terms, join paths, time semantics |
| Retrieval | [src/retrieval.py](src/retrieval.py) | Optional retrieval of relevant schema and few-shot examples (Chroma + OpenAI embeddings) |
| Text-to-SQL | [src/text_to_sql.py](src/text_to_sql.py) | `generate_sql` (uses schema + optional retrieval) |
| Validation | [src/sql_validator.py](src/sql_validator.py) | Syntax, safety, schema checks |
| Execution | [src/executor.py](src/executor.py) | Run SQL against SQLite |
| Answer generation | [src/llm_client.py](src/llm_client.py) | `generate_answer` (OpenAI/Anthropic with same fallback logic) |
| Pipelines | [src/pipeline.py](src/pipeline.py), [src/pipeline_langgraph.py](src/pipeline_langgraph.py) | Linear pipeline and LangGraph agent |
| App | [app.py](app.py) | Streamlit UI (calls normal pipeline) |
| Evaluation | [tools/run_eval.py](tools/run_eval.py) | Run benchmark or test set, normal or LangGraph; [evaluation/benchmark.json](evaluation/benchmark.json), [evaluation/test_set.json](evaluation/test_set.json) |

## Configuration and models

- **Current defaults (in `src/config.py`):** SQL and answer generation use `gpt-4o-mini`; fallback `claude-sonnet-4-20250514`.
- **Evaluation:** Running `python tools/run_eval.py` writes `evaluation/eval_results.json` with a `models` block so you can see which models were used.

Edit `src/config.py` to customize: `DEFAULT_SQL_MODEL`, `DEFAULT_ANSWER_MODEL`, `FALLBACK_SQL_MODEL`, `FALLBACK_ANSWER_MODEL`, `MAX_ROWS_FOR_ANSWER`, `MAX_SQL_RETRIES`.

## How It Works

1. **Load** – Excel files are loaded into pandas DataFrames and an in-memory SQLite DB.
2. **Knowledge injection** – Four types of knowledge are injected before SQL generation: structural, relational, semantic, and usage constraints (see [src/schema.py](src/schema.py)).
3. **Text-to-SQL** – The LLM maps the question to a SQL query (with optional retrieval and few-shot examples).
4. **Validate** – SQL is validated (syntax, safety, schema) before execution.
5. **Retry** – If validation fails, the LLM is asked to fix the SQL (up to 3 attempts).
6. **Execute** – The query runs against the SQLite DB.
7. **Answer** – The LLM generates a grounded answer from the retrieved data (row limit: 500).
8. **Fallback** – On API failure, the system retries with the fallback model; if answer generation fails, it returns raw data.

## ReAct & Chain of Thought (CoT)

- **CoT (Chain of Thought)** – The model is prompted to reason step-by-step before writing SQL: which tables/columns, join path, filters, aggregations.
- **ReAct (Reasoning + Acting)** – The model reasons from tool results (e.g. validation errors), then refines the SQL. It iteratively fixes invalid queries.
- **Linear pipeline** – CoT/ReAct appear in the schema prompt and in the retry message when validation fails.
- **LangGraph pipeline** – The agent uses CoT before each action and ReAct after each tool result (especially after `validate_sql` errors).

## Customizing Domain Knowledge

Edit [src/domain_knowledge.py](src/domain_knowledge.py) to add or modify: `TABLE_SEMANTICS`, `COLUMN_SEMANTICS`, `COUNTRY_REGIONS`, `REVENUE_AND_BILLING_TERMS`, `ALLOWED_JOIN_PATHS`, `TIME_SEMANTICS`, `FEW_SHOT_EXAMPLES`.

## Evaluation

- **Evaluation dataset:** Questions and expected SQL (and, for the benchmark, expected answers) in [evaluation/benchmark.json](evaluation/benchmark.json) and [evaluation/test_set.json](evaluation/test_set.json). The benchmark is built from data (no LLM) via [tools/build_benchmark.py](tools/build_benchmark.py).
- **What is evaluated:** (1) **SQL generation:** valid SQL rate, execution accuracy (result match), exact SQL match. (2) **Answer similarity** (benchmark only): agent answer vs ground truth (OpenAI embeddings, threshold 0.75). Domain knowledge (e.g. revenue/tax) and Chain-of-Thought/ReAct in the SQL prompt improve reasoning and error recovery. The answer-generator prompt is designed so the model answers from the given rows and only states that information is missing when it truly is.
- **Run eval:**
  ```bash
  python tools/run_eval.py                    # benchmark, normal pipeline
  python tools/run_eval.py --test-set         # 15 test questions, normal pipeline
  python tools/run_eval.py --test-set --langgraph   # 15 test questions, LangGraph
  python tools/run_eval.py --export-test-csv   # CSV: question, sql, agent_final_response
  ```
  Results: `evaluation/eval_results.json`, `evaluation/test_set_eval_results.json`, `evaluation/test_set_results.csv`.

## Assumptions

- **Reliability and fallback:** At each step (SQL generation, answer generation) there is a fallback: if the primary API (OpenAI) fails, Anthropic is used; if SQL validation fails, retries with error feedback are used; if the system cannot produce valid SQL after retries, it returns an error instead of executing invalid SQL.
- **Domain knowledge reflects business requirements:** Table/column semantics, join paths, time and revenue/tax rules are curated so the agent operates in the intended business context.
- **Read-only, controlled SQL:** Only SELECT (and WITH for CTEs) is allowed; write/admin operations are forbidden by the validator.

## Limitations

Right now we hand-write domain knowledge for each table (Clients, Invoices, InvoiceLineItems). That’s fine for a small schema, but if we add many more tables we’ll need a better approach—for example, figuring out structure and relationships from the schema or metadata instead of maintaining it all by hand.

The system is built for this business context. Using it in a different domain would mean reworking the domain knowledge and likely the prompts.

We send a lot of context to the model (schema, few-shot examples, CoT/ReAct), so API usage can get costly. We could shrink the prompt or lean more on retrieval to save cost, but that might hurt accuracy.

We rely on external APIs (OpenAI, Anthropic), so data goes out of our environment and we depend on their availability and policies. If you need things fully in-house or offline, you’d have to switch to something like local models.

When the SQL result is fuzzy or the “answer” isn’t just a single fact in the table, the answer generator has to do more interpretation. We’d need more prompt tuning (and maybe more evaluation) to get “do we have enough to answer?” right in those cases.

A few more constraints: we only use in-memory SQLite, we cap at 500 rows when generating the natural-language answer, and the test set doesn’t have ground-truth answers—so we only compute answer similarity on the benchmark.
