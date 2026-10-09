"""Evaluation utilities for the Text-to-SQL agent.

- run_sql: read-only SQLite execution with a timeout
- score: three result-comparison variants (all column-order sensitive)
- wilson_ci / paired_report: confidence intervals and paired significance tests
- CachedLLM: disk-cached LLM calls with rate-limit backoff and a token budget
- run_system / summarize / compare: resumable evaluation harness
"""
import hashlib
import json
import os
import re
import sqlite3
import time
from collections import Counter

import numpy as np

MAX_ROWS = 50_000


# ---------------------------------------------------------------- execution
def run_sql(db_path, sql, timeout_s=10.0):
    """Execute SQL read-only with a timeout. Returns (ok, rows_or_error_string)."""
    conn = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        deadline = time.time() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.time() > deadline else 0, 20000)
        rows = conn.execute(sql).fetchmany(MAX_ROWS)
        return True, rows
    except Exception as e:
        return False, str(e)
    finally:
        if conn is not None:
            conn.close()


# ------------------------------------------------------------------ scoring
def _norm_rows(rows):
    return [tuple(round(v, 6) if isinstance(v, float) else v for v in r) for r in rows]


def _top_level_order_by(sql):
    depth, out = 0, []
    for ch in sql:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return bool(re.search(r"\border\s+by\b", "".join(out), re.I))


def score(gold_sql, gold_rows, pred_rows):
    """All variants compare full rows, so column order matters.
    legacy   : set of rows equal (the metric used for every number reported so far)
    multiset : same rows with the same multiplicities
    ordered  : like multiset, but row order must also match when the gold SQL has a top-level ORDER BY
    """
    g, p = _norm_rows(gold_rows), _norm_rows(pred_rows)
    multiset = Counter(g) == Counter(p)
    ordered = (g == p) if _top_level_order_by(gold_sql) else multiset
    return {"legacy": set(g) == set(p), "multiset": multiset, "ordered": ordered}


def approx_hardness(sql):
    """Heuristic difficulty label (NOT the official Spider classifier)."""
    s = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql.lower())
    comp = sum(bool(re.search(p, s)) for p in
               [r"\bjoin\b", r"\bgroup\s+by\b", r"\bhaving\b", r"\border\s+by\b", r"\blimit\b", r"\blike\b"])
    nested = bool(re.search(r"\b(intersect|union|except)\b|\(\s*select\b", s))
    if nested:
        return "hard" if comp <= 2 else "extra"
    if comp == 0:
        return "easy"
    return "medium" if comp <= 2 else "hard"


# --------------------------------------------------------------- statistics
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, center - half), min(1.0, center + half)


def paired_report(a, b, n_boot=5000, seed=0):
    """a, b: aligned boolean lists (system A, system B) over the same questions."""
    from scipy.stats import binomtest
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    fixed, broken = int((~a & b).sum()), int((a & ~b).sum())
    p = binomtest(fixed, fixed + broken, 0.5).pvalue if fixed + broken else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    diffs = b[idx].mean(1) - a[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"acc_a": float(a.mean()), "acc_b": float(b.mean()), "diff": float(b.mean() - a.mean()),
            "diff_ci95": (float(lo), float(hi)), "b_fixed": fixed, "b_broke": broken, "p_value": float(p)}


# ---------------------------------------------------------------- LLM cache
class BudgetExceeded(Exception):
    """Raised when the token budget or a long/daily rate limit stops further calls."""


def _retry_after_seconds(msg):
    m = re.search(r"try again in ([0-9hms.\s]+)", msg)
    if not m:
        return None
    total = 0.0
    for num, unit in re.findall(r"([\d.]+)(ms|h|m|s)", m.group(1)):
        total += float(num) * {"ms": 0.001, "s": 1, "m": 60, "h": 3600}[unit]
    return total


class CachedLLM:
    """Wraps a Groq client: identical requests are answered from a JSONL cache (costs zero tokens)."""

    def __init__(self, client, cache_path, model="openai/gpt-oss-120b", token_budget=None, max_retries=6):
        self.client, self.model = client, model
        self.cache_path, self.token_budget, self.max_retries = cache_path, token_budget, max_retries
        self.cache, self.tokens_used, self.hits, self.calls = {}, 0, 0, 0
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.cache[rec["k"]] = rec["text"]
                    except Exception:
                        pass

    def _key(self, messages, **params):
        blob = json.dumps({"model": self.model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def chat(self, messages, temperature=0, max_tokens=None, tag=""):
        key = self._key(messages, temperature=temperature, max_tokens=max_tokens, tag=tag)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        if self.token_budget is not None and self.tokens_used >= self.token_budget:
            raise BudgetExceeded(f"Token budget reached ({self.tokens_used}/{self.token_budget}).")
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kw = dict(model=self.model, messages=messages, temperature=temperature)
                if max_tokens:
                    kw["max_completion_tokens"] = max_tokens
                resp = self.client.chat.completions.create(**kw)
                self.calls += 1
                self.tokens_used += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
                text = (resp.choices[0].message.content or "").strip()
                self.cache[key] = text
                with open(self.cache_path, "a") as f:
                    f.write(json.dumps({"k": key, "text": text}) + "\n")
                return text
            except Exception as e:
                msg = str(e)
                if "429" not in msg and "rate_limit" not in msg:
                    raise
                last_err = msg
                wait = _retry_after_seconds(msg)
                if wait is not None and wait > 90:
                    raise BudgetExceeded(f"Rate limit needs a {wait/60:.0f} min wait (likely the daily cap): {msg[:160]}")
                time.sleep((wait if wait is not None else 2 ** attempt) + 0.5)
        raise BudgetExceeded(f"Still rate limited after {self.max_retries} retries: {str(last_err)[:160]}")


# ------------------------------------------------------------------ harness
def run_system(name, system_fn, examples, db_path_fn, out_path, verbose=True):
    """Run system_fn(example) -> {'pred_sql': ..., ...extra} over examples.
    Results append to a JSONL file after every example, so a rerun resumes where it stopped.
    Rate-limit / budget stops end the run cleanly WITHOUT recording the question as a failure."""
    done = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    todo = [e for e in examples if e["id"] not in done]
    if verbose:
        print(f"[{name}] {len(done)} already done, {len(todo)} to run")
    with open(out_path, "a") as f:
        for i, ex in enumerate(todo):
            path = db_path_fn(ex)
            gold_ok, gold_rows = run_sql(path, ex["query"])
            t0, err = time.time(), None
            try:
                out = system_fn(ex)
            except BudgetExceeded as e:
                print(f"[{name}] stopping early: {e}")
                break
            except Exception as e:
                out, err = {"pred_sql": ""}, f"{type(e).__name__}: {e}"
            pred_sql = (out.get("pred_sql") or "").strip()
            pred_ok, pred_rows = run_sql(path, pred_sql) if pred_sql else (False, "empty prediction")
            sc = score(ex["query"], gold_rows, pred_rows) if (gold_ok and pred_ok) else \
                {"legacy": False, "multiset": False, "ordered": False}
            rec = {"id": ex["id"], "db_id": ex["db_id"], "question": ex["question"], "gold_sql": ex["query"],
                   "hardness": approx_hardness(ex["query"]), "pred_sql": pred_sql, "executes": pred_ok,
                   "gold_executes": gold_ok, "error": err or (None if pred_ok else str(pred_rows)),
                   "extra": {k: v for k, v in out.items() if k != "pred_sql"},
                   "seconds": round(time.time() - t0, 2), **sc}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[ex["id"]] = rec
            if verbose and (i + 1) % 10 == 0:
                print(f"[{name}] {i + 1}/{len(todo)} new results")
    return [done[e["id"]] for e in examples if e["id"] in done]


def summarize(results, label=""):
    n = len(results)
    if n == 0:
        print(f"{label}: no results")
        return
    k = sum(r["legacy"] for r in results)
    lo, hi = wilson_ci(k, n)
    valid = sum(r["executes"] for r in results)
    print(f"{label}: {k}/{n} = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%}) | valid-SQL {valid / n:.1%}")
    print("   other scorers: multiset {:.1%} | ordered {:.1%}".format(
        sum(r["multiset"] for r in results) / n, sum(r["ordered"] for r in results) / n))
    by = {}
    for r in results:
        by.setdefault(r["hardness"], []).append(r["legacy"])
    print("   by approx. hardness: " + " | ".join(
        f"{h} {sum(by[h])}/{len(by[h])}" for h in ["easy", "medium", "hard", "extra"] if h in by))


def compare(results_a, results_b, label_a="A", label_b="B", metric="legacy"):
    bm = {r["id"]: r for r in results_b}
    pairs = [(r, bm[r["id"]]) for r in results_a if r["id"] in bm]
    rep = paired_report([a[metric] for a, _ in pairs], [b[metric] for _, b in pairs])
    print(f"{label_a} vs {label_b} on {len(pairs)} shared questions ({metric}): "
          f"{rep['acc_a']:.1%} -> {rep['acc_b']:.1%} (diff {rep['diff']:+.1%}, 95% CI "
          f"{rep['diff_ci95'][0]:+.1%} to {rep['diff_ci95'][1]:+.1%}) | {label_b} fixed {rep['b_fixed']}, "
          f"broke {rep['b_broke']}, p = {rep['p_value']:.3f}")
    return rep
