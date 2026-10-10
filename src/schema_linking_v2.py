"""Schema linking v2.

Compared with v1 (one embedding per table, top-k tables) this adds:
  1. column-level embeddings (a table scores on its name AND its best-matching columns)
  2. value linking: literals in the question ('Kaloyan', 'Republic') are matched against the
     actual cell values of text columns, boosting the table/column that holds them
  3. foreign-key bridging: tables that connect two selected tables (e.g. singer_in_concert)
     are added, because a join path is useless with a missing link table
  4. column pruning for wide tables (keeps PK/FK/value-matched/most relevant columns)
  5. schema text with proper composite primary keys and a hint line listing value matches
"""
import itertools
import re
import sqlite3
from collections import defaultdict, deque

import numpy as np

STOP = set("""a an the of in on at to for from by with and or is are was were be been what which who whom whose
how many much list show give find tell all each every their there that this these those me us do does did have has
had than then also not no name names number id""".split())


def _words(identifier):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", identifier)
    return re.sub(r"[_\s]+", " ", s).strip().lower()


def _norm_text(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _is_text_type(t):
    t = (t or "").lower()
    return any(k in t for k in ("char", "text", "clob", "varchar")) or t == ""


class SchemaIndex:
    def __init__(self, db_path, embedder, max_values_per_col=5000, max_value_len=60):
        self.db_path, self.embedder = db_path, embedder
        self.tables = {}          # table -> {'columns': [{name,type,pk_order}], 'fks': [...], 'sample_rows': [...]}
        self.value_index = defaultdict(set)   # normalized value -> {(table, column)}
        self._load(max_values_per_col, max_value_len)
        self._embed()

    # ------------------------------------------------------------ loading
    def _load(self, max_values_per_col, max_value_len):
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cur = conn.cursor()
        names = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for t in names:
            cols = [{"name": c[1], "type": c[2], "pk_order": c[5]} for c in cur.execute(f"PRAGMA table_info('{t}')")]
            fks = [{"from": f[3], "to_table": f[2], "to_col": f[4]} for f in cur.execute(f"PRAGMA foreign_key_list('{t}')")]
            try:
                samples = cur.execute(f'SELECT * FROM "{t}" LIMIT 3').fetchall()
            except Exception:
                samples = []
            self.tables[t] = {"columns": cols, "fks": fks, "sample_rows": samples}
            for c in cols:
                c["examples"] = []
                if not _is_text_type(c["type"]):
                    continue
                try:
                    vals = [r[0] for r in cur.execute(
                        f'SELECT DISTINCT "{c["name"]}" FROM "{t}" WHERE "{c["name"]}" IS NOT NULL LIMIT {max_values_per_col}')]
                except Exception:
                    continue
                for v in vals:
                    if not isinstance(v, str) or len(v) > max_value_len:
                        continue
                    nv = _norm_text(v)
                    if len(nv) < 3 or nv.replace(" ", "").isdigit():
                        continue
                    self.value_index[nv].add((t, c["name"]))
                    if len(c["examples"]) < 3:
                        c["examples"].append(v)
        conn.close()
        # undirected FK graph (only edges whose target table exists)
        self.graph = defaultdict(set)
        for t, info in self.tables.items():
            for fk in info["fks"]:
                if fk["to_table"] in self.tables:
                    self.graph[t].add(fk["to_table"])
                    self.graph[fk["to_table"]].add(t)

    def _embed(self):
        self.table_names = list(self.tables)
        t_docs = [f"table {_words(t)} with columns " + ", ".join(_words(c["name"]) for c in self.tables[t]["columns"])
                  for t in self.table_names]
        self.col_keys, c_docs = [], []
        for t in self.table_names:
            for c in self.tables[t]["columns"]:
                ex = f", values like {', '.join(c['examples'])}" if c["examples"] else ""
                c_docs.append(f"column {_words(c['name'])} of table {_words(t)}{ex}")
                self.col_keys.append((t, c["name"]))
        self.table_emb = self._enc(t_docs)
        self.col_emb = self._enc(c_docs) if c_docs else np.zeros((0, self.table_emb.shape[1]))
        self.col_owner = np.array([self.table_names.index(t) for t, _ in self.col_keys], dtype=int)

    def _enc(self, docs):
        e = np.asarray(self.embedder.encode(docs, convert_to_numpy=True), dtype=np.float32)
        return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)

    # ------------------------------------------------------- value linking
    def match_values(self, question, max_matches=8):
        toks = _norm_text(question).split()
        found = {}
        for n in range(min(6, len(toks)), 0, -1):
            for i in range(len(toks) - n + 1):
                gram = " ".join(toks[i:i + n])
                if gram not in self.value_index:
                    continue
                if n == 1 and (gram in STOP or len(gram) < 4):
                    continue
                if any(gram in longer for longer in found if longer != gram):
                    continue          # already covered by a longer matched phrase
                found[gram] = sorted(self.value_index[gram])
        out = []
        for gram, locs in found.items():
            for (t, c) in locs:
                out.append({"value": gram, "table": t, "column": c})
        return out[:max_matches]

    # -------------------------------------------------------------- bridging
    def _shortest_path(self, a, b, max_edges=3):
        q, seen = deque([[a]]), {a}
        while q:
            path = q.popleft()
            if path[-1] == b:
                return path
            if len(path) - 1 >= max_edges:
                continue
            for nb in sorted(self.graph[path[-1]]):      # sorted: deterministic across runs
                if nb not in seen:
                    seen.add(nb)
                    q.append(path + [nb])
        return None

    def _bridge(self, selected, max_extra=2):
        extra = []
        for a, b in itertools.combinations(selected, 2):
            path = self._shortest_path(a, b)
            if path and len(path) > 2:
                for node in path[1:-1]:
                    if node not in selected and node not in extra:
                        extra.append(node)
        return extra[:max_extra]

    # ----------------------------------------------------------------- link
    def link(self, question, top_k=4, max_cols=12, use_values=True, use_bridge=True,
             full_threshold=4, value_bonus=0.3):
        q = self._enc([question])[0]
        t_sims = self.table_emb @ q
        c_sims = self.col_emb @ q if len(self.col_keys) else np.zeros(0)
        best_col = np.zeros(len(self.table_names))
        for i in range(len(self.table_names)):
            mask = self.col_owner == i
            best_col[i] = c_sims[mask].max() if mask.any() else 0.0
        scores = 0.5 * t_sims + 0.5 * best_col

        matches = self.match_values(question) if use_values else []
        matched_cols = defaultdict(set)
        for m in matches:
            matched_cols[m["table"]].add(m["column"])
        for t, cols in matched_cols.items():
            scores[self.table_names.index(t)] += min(2 * value_bonus, value_bonus * len(cols))

        order = [self.table_names[i] for i in np.argsort(-scores)]
        selected = order if len(order) <= full_threshold else order[:top_k]
        bridged = self._bridge(selected) if (use_bridge and len(order) > full_threshold) else []
        selected = selected + bridged

        col_sim = {k: float(c_sims[i]) for i, k in enumerate(self.col_keys)}
        keep = {}
        for t in selected:
            cols = self.tables[t]["columns"]
            if len(cols) <= max_cols:
                keep[t] = [c["name"] for c in cols]
                continue
            must = {c["name"] for c in cols if c["pk_order"] > 0}
            must |= {fk["from"] for fk in self.tables[t]["fks"]}
            must |= matched_cols.get(t, set())
            rest = sorted((c["name"] for c in cols if c["name"] not in must),
                          key=lambda n: -col_sim.get((t, n), 0.0))
            chosen = must | set(rest[:max(0, max_cols - len(must))])
            keep[t] = [c["name"] for c in cols if c["name"] in chosen]

        return {"tables": selected, "bridged": bridged, "columns": keep, "value_matches": matches,
                "scores": {t: float(scores[self.table_names.index(t)]) for t in selected},
                "schema_text": self.schema_text(selected, keep, matches)}

    # ------------------------------------------------------------ rendering
    def schema_text(self, tables, keep, matches=(), include_samples=False):
        wanted = set(tables)
        tables = [t for t in self.table_names if t in wanted]   # canonical order: same as in the database
        blocks = []
        for t in tables:
            info = self.tables[t]
            kept = set(keep.get(t) or [c["name"] for c in info["columns"]])   # no columns given -> show all
            lines = []
            for c in info["columns"]:
                if c["name"] in kept:
                    pk = " PRIMARY KEY" if (c["pk_order"] > 0 and sum(x["pk_order"] > 0 for x in info["columns"]) == 1) else ""
                    lines.append(f"  {c['name']} {c['type']}{pk}")
            pks = [c["name"] for c in sorted((c for c in info["columns"] if c["pk_order"] > 0), key=lambda c: c["pk_order"])]
            if len(pks) > 1 and all(p in kept for p in pks):
                lines.append(f"  PRIMARY KEY ({', '.join(pks)})")
            for fk in info["fks"]:
                if fk["from"] in kept and fk["to_table"] in wanted:
                    lines.append(f"  FOREIGN KEY ({fk['from']}) REFERENCES {fk['to_table']}({fk['to_col']})")
            block = f"CREATE TABLE {t} (\n" + ",\n".join(lines) + "\n);"
            if include_samples and info["sample_rows"]:
                idx = [i for i, c in enumerate(info["columns"]) if c["name"] in kept]
                block += f"\n/* sample rows of {t}: " + "; ".join(
                    str(tuple(r[i] for i in idx)) for r in info["sample_rows"][:2]) + " */"
            blocks.append(block)
        text = "\n\n".join(blocks)
        if matches:
            hints = "; ".join(f"'{m['value']}' appears in {m['table']}.{m['column']}" for m in matches)
            text += f"\n\n/* Values mentioned in the question: {hints} */"
        return text


# ------------------------------------------------- gold-schema recall metrics
def gold_schema_elements(sql, tables):
    """Approximate gold tables/columns by identifier matching (tables: {table: [column names]}).
    Over-marks when a column name is shared by several used tables, so recall is slightly pessimistic."""
    s = re.sub(r"'[^']*'|\"[^\"]*\"", " ", sql.lower())
    toks = set(re.findall(r"[a-z_][a-z0-9_]*", s))
    used = {t: [c for c in cols if c.lower() in toks] for t, cols in tables.items() if t.lower() in toks}
    return used


def link_recall(result_tables, result_columns, gold):
    """result_*: what the linker kept; gold: {table: [cols]} -> (all_tables_covered, cols_found, cols_total)"""
    covered = all(t in result_tables for t in gold)
    found = total = 0
    for t, cols in gold.items():
        for c in cols:
            total += 1
            if t in result_columns and c in result_columns[t]:
                found += 1
    return covered, found, total


# ------------------------------------------------------------- benchmark
def benchmark_linking(examples, db_path_fn, embedder, save_path=None, show_misses=8):
    """Compare v1 (table-level) and v2 linking against the tables/columns the gold SQL uses.
    Needs no LLM calls and no GPU. examples: dicts with db_id, question, query."""
    import json
    from schema_linking import build_table_documents, retrieve_relevant_tables   # the v1 module

    v2_cfgs = {
        "v2 embeddings only, k=3": dict(top_k=3, use_values=False, use_bridge=False),
        "v2 + values, k=3": dict(top_k=3, use_values=True, use_bridge=False),
        "v2 + values + bridge, k=3": dict(top_k=3, use_values=True, use_bridge=True),
        "v2 + values + bridge, k=4": dict(top_k=4, use_values=True, use_bridge=True),
    }
    names = (["full schema (no linking)"] + [f"v1 table-level, k={k}" for k in (2, 3, 4)] + list(v2_cfgs))
    acc = {n: dict(cov=0, found=0, total=0, sel=0, frac=0.0) for n in names}
    misses, indexes, used, skipped = [], {}, 0, 0

    for i, ex in enumerate(examples):
        if ex["db_id"] not in indexes:
            indexes[ex["db_id"]] = SchemaIndex(db_path_fn(ex), embedder)
        idx = indexes[ex["db_id"]]
        all_cols = {t: [c["name"] for c in idx.tables[t]["columns"]] for t in idx.tables}
        gold = gold_schema_elements(ex["query"], all_cols)
        if not gold:
            skipped += 1
            continue
        used += 1
        v1_names, v1_docs = build_table_documents(idx.tables)
        outputs = {"full schema (no linking)": (list(all_cols), all_cols)}
        for k in (2, 3, 4):
            sel = [t for t, _ in retrieve_relevant_tables(ex["question"], v1_names, v1_docs, embedder, top_k=k)]
            outputs[f"v1 table-level, k={k}"] = (sel, {t: all_cols[t] for t in sel})
        for n, cfg in v2_cfgs.items():
            r = idx.link(ex["question"], **cfg)
            outputs[n] = (r["tables"], r["columns"])
        for n, (tabs, cols) in outputs.items():
            cov, found, total = link_recall(tabs, cols, gold)
            a = acc[n]
            a["cov"] += cov
            a["found"] += found
            a["total"] += total
            a["sel"] += len(tabs)
            a["frac"] += len(tabs) / len(all_cols)
        cov, _, _ = link_recall(*outputs["v2 + values + bridge, k=3"], gold)
        if not cov and len(misses) < show_misses:
            misses.append({"question": ex["question"], "db": ex["db_id"], "gold_tables": sorted(gold),
                           "selected": outputs["v2 + values + bridge, k=3"][0]})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(examples)} questions processed ({len(indexes)} databases indexed)")

    rows = []
    print(f"\n{used} questions scored ({skipped} skipped: gold tables not recognised)\n")
    print(f"{'config':32s} {'avg tables':>10s} {'% of schema':>11s} {'all tables found':>17s} {'column recall':>14s}")
    for n in names:
        a = acc[n]
        row = {"config": n, "avg_tables": a["sel"] / used, "schema_fraction": a["frac"] / used,
               "all_gold_tables_found": a["cov"] / used, "column_recall": a["found"] / max(1, a["total"])}
        rows.append(row)
        print(f"{n:32s} {row['avg_tables']:10.2f} {row['schema_fraction']:11.0%} "
              f"{row['all_gold_tables_found']:17.1%} {row['column_recall']:14.1%}")
    print("\nSample misses for 'v2 + values + bridge, k=3':")
    for m in misses:
        print(f"  [{m['db']}] {m['question']}\n      gold tables: {m['gold_tables']} | selected: {m['selected']}")
    if save_path:
        with open(save_path, "w") as f:
            json.dump({"n_questions": used, "rows": rows, "misses": misses}, f, indent=2)
    return rows
