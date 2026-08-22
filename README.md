# Enterprise Text-to-SQL Agent

A schema-linking, self-correcting Text-to-SQL agent built on prompting strategies from **DIN-SQL** (Pourreza et al.) and **DAIL-SQL** (Gao et al.). Given a natural language question and a database, it retrieves the relevant schema, classifies query difficulty, generates SQL, and self-corrects on execution failure.

**🔗 Live demo:** [huggingface.co/spaces/korg-0/text2sql-agent](https://huggingface.co/spaces/korg-0/text2sql-agent)

## The Problem

Generic LLM prompting for SQL generation struggles on multi-table enterprise databases: dumping a full schema into every prompt wastes context and increases hallucinated column/table names, and single-shot generation has no mechanism to recover from syntax or logic errors.

## Approach

1. **Schema linking** — each table is embedded (`all-MiniLM-L6-v2`) alongside the question; cosine similarity retrieves only the top-k most relevant tables, so the LLM never sees irrelevant schema.
2. **Difficulty classification** — following DIN-SQL, each question is classified as `EASY` / `NON-NESTED` / `NESTED`, which adjusts the reasoning instruction given to the generator.
3. **Few-shot SQL generation** — the model (`openai/gpt-oss-120b` via Groq) generates SQL using difficulty-aware few-shot examples (DAIL-SQL's core finding: well-chosen examples materially improve accuracy over zero-shot).
4. **Execution-based self-correction** — generated SQL is executed against the actual SQLite database; on failure, the error message is fed back to the model for up to 3 correction attempts.

## Results

Evaluated on a sample of the Spider benchmark validation set (execution accuracy — generated SQL's result set compared against gold SQL's result set):

| Run | Examples | Accuracy |
|---|---|---|
| Baseline | 50 | 82.0% |
| After prompt fix (exact-match enforcement) | 44* | 90.9% |

\* 6 examples excluded from the second run due to free-tier API rate limiting, not model failure.

**Iteration note:** the baseline run showed a clear failure pattern — the model defaulting to `LIKE '%...%'` wildcard matching where exact equality was expected (e.g. matching `"Republic"` government forms, or exact airport names). Adding an explicit instruction against unnecessary wildcard matching fixed 3 of 9 original failures.

### Known limitations
- Semantic ambiguity in aggregation edge cases (e.g. `INNER` vs `LEFT JOIN` interpretation of "least X")
- Set-operation equivalents (`EXCEPT` vs `NOT EXISTS`/anti-join) occasionally diverge from gold query structure despite comparable intent
- The agent does not currently verify a question is answerable from the selected database's schema before attempting generation — mismatched database/question pairs can produce degenerate output

## Architecture
├── app/
│ ├── app.py # Gradio interface
│ └── requirements.txt
├── src/
│ ├── schema_linking.py # schema extraction + embedding-based retrieval
│ └── sql_generation.py # difficulty classification, generation, self-correction
├── evaluation/
│ └── eval_results.json # benchmark results
└── spider_databases/ # SQLite databases (Spider validation split, 20 DBs)

## Tech Stack

- **LLM inference:** Groq API (`openai/gpt-oss-120b`)
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Benchmark:** [Spider](https://yale-lily.github.io/spider) (validation split, 20 databases)
- **Interface:** Gradio, deployed on Hugging Face Spaces
- **Dev environment:** Google Colab

## Running Locally

```bash
pip install -r app/requirements.txt
export GROQ_API_KEY=your_key_here
python app/app.py
```

## References

- Pourreza, M., & Rafiei, D. (2023). [DIN-SQL: Decomposed In-Context Learning of Text-to-SQL with Self-Correction](https://arxiv.org/abs/2304.11015)
- Gao, D. et al. (2023). [Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation](https://arxiv.org/abs/2308.15363) (DAIL-SQL)
- Yu, T. et al. (2018). [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
