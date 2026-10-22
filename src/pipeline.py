"""LLM pipeline (Groq): schema linking -> optional sub-question decomposition -> SQL generation
-> execution-guided repair.

mode="direct"     : link, generate, repair            (no planning call; the ablation baseline)
mode="decompose"  : link, PLAN (difficulty + sub-questions), generate with the sub-questions, repair

Prompts put the fixed instructions first and the schema/question last so repeated prefixes can be cached.
All LLM calls go through evaluation.CachedLLM (disk cache, backoff, token budget).
"""
import json
import re

from evaluation import run_sql

RULES = """Rules:
- Write one SQLite query. Use ONLY tables and columns that appear in the schema.
- Use exact equality (=) for text comparisons unless the question clearly asks for partial matching.
- Return exactly the columns the question asks for, in the order asked.
- Prefer JOIN + GROUP BY/HAVING; use a subquery or INTERSECT/EXCEPT/UNION only when the question needs one."""

GEN_SYSTEM = ("You are an expert SQLite query writer.\n" + RULES +
              "\nRespond with ONLY the SQL query: no explanation, no markdown, no backticks.")

PLAN_SYSTEM = ('You plan SQL queries. Given a schema and a question, reply with ONLY a JSON object:\n'
               '{"difficulty": "EASY" | "NON-NESTED" | "NESTED", "sub_questions": ["...", "..."]}\n'
               "EASY = one table, no join. NON-NESTED = joins and/or GROUP BY/HAVING, no subquery. "
               "NESTED = needs a subquery or INTERSECT/UNION/EXCEPT.\n"
               "For EASY use an empty list. Otherwise split the question into at most 4 short, ordered "
               "sub-questions that can each be answered from the schema; the last one produces the final answer.")


def clean_sql(text):
    return re.sub(r"```(?:sql)?", "", text or "").strip()


def parse_plan(text):
    """Tolerant JSON parsing; falls back to NON-NESTED with no sub-questions."""
    difficulty, subs = "NON-NESTED", []
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            d = str(obj.get("difficulty", "")).upper().replace("_", "-").strip()
            if d in ("EASY", "NON-NESTED", "NESTED"):
                difficulty = d
            subs = [str(x).strip() for x in obj.get("sub_questions", []) if str(x).strip()][:4]
        except Exception:
            pass
    if difficulty == "EASY":
        subs = []
    return difficulty, subs


def gen_messages(schema, question, sub_questions):
    user = f"Schema:\n{schema}\n\nQuestion: {question}"
    if sub_questions:
        steps = "\n".join(f"{i}. {s}" for i, s in enumerate(sub_questions, 1))
        user += f"\n\nWork through these sub-questions in order, then write the single final query:\n{steps}"
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]


def repair_messages(schema, question, sql, error):
    user = (f"Schema:\n{schema}\n\nQuestion: {question}\n\nThis query failed in SQLite:\n{sql}\n\n"
            f"Error: {error}\n\nFix it using only tables and columns from the schema. Respond with ONLY the corrected SQL.")
    return [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}]


class SQLPipeline:
    def __init__(self, llm, embedder, db_path_fn, mode="decompose", k=4, full_threshold=8,
                 use_values=False, max_repairs=2):
        assert mode in ("direct", "decompose")
        self.llm, self.embedder, self.db_path_fn = llm, embedder, db_path_fn
        self.mode, self.k, self.full_threshold = mode, k, full_threshold
        self.use_values, self.max_repairs = use_values, max_repairs
        self._indexes = {}

    def _index(self, path):
        from schema_linking_v2 import SchemaIndex
        if path not in self._indexes:
            self._indexes[path] = SchemaIndex(path, self.embedder)
        return self._indexes[path]

    def __call__(self, ex):
        path = self.db_path_fn(ex)
        link = self._index(path).link(ex["question"], top_k=self.k, full_threshold=self.full_threshold,
                                      use_values=self.use_values, use_bridge=True)
        schema = link["schema_text"]
        difficulty, subs = "n/a", []
        if self.mode == "decompose":
            plan = self.llm.chat([{"role": "system", "content": PLAN_SYSTEM},
                                  {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {ex['question']}"}],
                                 tag="plan")
            difficulty, subs = parse_plan(plan)
        sql = clean_sql(self.llm.chat(gen_messages(schema, ex["question"], subs), tag="gen"))
        repairs = 0
        while repairs < self.max_repairs:
            ok, res = run_sql(path, sql)
            if ok:
                break
            repairs += 1
            sql = clean_sql(self.llm.chat(repair_messages(schema, ex["question"], sql, res), tag=f"repair{repairs}"))
        return {"pred_sql": sql, "difficulty": difficulty, "sub_questions": subs,
                "repairs": repairs, "linked_tables": link["tables"]}
