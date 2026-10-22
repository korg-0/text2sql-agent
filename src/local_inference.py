"""Local-model inference helpers.

- PromptBuilder: builds the exact prompt format each model was trained on
    v1 -> full schema in the v1 layout (no sample rows)
    v2 -> linked schema from schema_linking_v2 (K tables + bridges, tables in database order)
- hf_generate: greedy or sampled generation with a loaded Hugging Face model (only function that needs torch)
- vote: execution-guided self-consistency (majority result among candidates that actually execute)
- make_local_system: wraps everything into the system_fn that evaluation.run_system expects
"""
import re

from evaluation import _norm_rows, run_sql

SYSTEM_PROMPT = ("You are an expert SQL generator for SQLite databases. Generate a single, syntactically correct "
                 "SQL query that answers the question. Respond with ONLY the SQL query, no explanation, no "
                 "markdown formatting, no backticks.")


def clean_sql(text):
    return re.sub(r"```(?:sql)?", "", text or "").strip()


def _messages(schema_text, question):
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Schema:\n{schema_text}\n\nQuestion: {question}"}]


class PromptBuilder:
    def __init__(self, embedder, k=4, full_threshold=8):
        self.embedder, self.k, self.full_threshold = embedder, k, full_threshold
        self._v1_cache, self._index_cache = {}, {}

    def v1(self, db_path, question):
        from schema_linking import get_db_schema, schema_to_text      # the original v1 module
        if db_path not in self._v1_cache:
            self._v1_cache[db_path] = schema_to_text(get_db_schema(db_path), include_samples=False)
        return _messages(self._v1_cache[db_path], question)

    def v2(self, db_path, question):
        from schema_linking_v2 import SchemaIndex
        if db_path not in self._index_cache:
            self._index_cache[db_path] = SchemaIndex(db_path, self.embedder)
        r = self._index_cache[db_path].link(question, top_k=self.k, full_threshold=self.full_threshold,
                                            use_values=False, use_bridge=True)
        return _messages(r["schema_text"], question)


def hf_generate(model, tokenizer, messages, max_new_tokens=256, n=1, temperature=0.0, top_p=0.95):
    """Returns a list of n SQL strings. temperature=0 -> one greedy answer."""
    import torch
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    if temperature and temperature > 0:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, num_return_sequences=n)
    else:
        kw.update(do_sample=False)
    with torch.no_grad():
        out = model.generate(**inputs, **kw)
    start = inputs["input_ids"].shape[1]
    return [clean_sql(tokenizer.decode(o[start:], skip_special_tokens=True)) for o in out]


def vote(db_path, greedy_sql, samples):
    """Group the greedy answer and the samples by their execution result; return the SQL from the
    biggest group (ties go to the group containing the greedy answer). If nothing executes, keep greedy."""
    groups = {}
    for sql in [greedy_sql] + list(samples):
        ok, rows = run_sql(db_path, sql)
        if ok:
            groups.setdefault(frozenset(_norm_rows(rows)), []).append(sql)
    if not groups:
        return greedy_sql, 0
    best = max(groups.values(), key=lambda g: (len(g), greedy_sql in g))
    return (greedy_sql if greedy_sql in best else best[0]), len(best)


def make_local_system(gen_fn, prompt_fn, db_path_fn, n_votes=0, temperature=0.7):
    """gen_fn(messages, n=1, temperature=0.0) -> list[str];  prompt_fn(example, db_path) -> messages."""
    def system(ex):
        path = db_path_fn(ex)
        messages = prompt_fn(ex, path)
        greedy = gen_fn(messages)[0]
        if n_votes <= 0:
            return {"pred_sql": greedy}
        samples = gen_fn(messages, n=n_votes, temperature=temperature)
        sql, agree = vote(path, greedy, samples)
        return {"pred_sql": sql, "greedy_sql": greedy, "votes_for_winner": agree}
    return system
