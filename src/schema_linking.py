
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

def get_db_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        foreign_keys = cursor.fetchall()
        try:
            cursor.execute(f"SELECT * FROM '{table}' LIMIT 3")
            sample_rows = cursor.fetchall()
        except Exception:
            sample_rows = []
        schema[table] = {
            'columns': [{'name': c[1], 'type': c[2], 'pk': bool(c[5])} for c in columns],
            'foreign_keys': [{'from': fk[3], 'to_table': fk[2], 'to_col': fk[4]} for fk in foreign_keys],
            'sample_rows': sample_rows
        }
    conn.close()
    return schema


def schema_to_text(schema, include_samples=True, tables_filter=None):
    output = []
    for table, info in schema.items():
        if tables_filter is not None and table not in tables_filter:
            continue
        lines = [f"CREATE TABLE {table} ("]
        col_lines = []
        for col in info['columns']:
            pk_marker = " PRIMARY KEY" if col['pk'] else ""
            col_lines.append(f"  {col['name']} {col['type']}{pk_marker}")
        for fk in info['foreign_keys']:
            col_lines.append(f"  FOREIGN KEY ({fk['from']}) REFERENCES {fk['to_table']}({fk['to_col']})")
        lines.append(",\n".join(col_lines))
        lines.append(");")
        table_sql = "\n".join(lines)
        if include_samples and info['sample_rows']:
            col_names = [c['name'] for c in info['columns']]
            sample_str = f"\n/* Sample rows from {table}:\n{col_names}\n"
            for row in info['sample_rows'][:2]:
                sample_str += f"{row}\n"
            sample_str += "*/"
            table_sql += sample_str
        output.append(table_sql)
    return "\n\n".join(output)


def build_table_documents(schema):
    documents = []
    table_names = []
    for table, info in schema.items():
        col_desc = ", ".join([c['name'] for c in info['columns']])
        documents.append(f"Table {table} contains columns: {col_desc}")
        table_names.append(table)
    return table_names, documents


def retrieve_relevant_tables(question, table_names, documents, embedder, top_k=3):
    doc_embeddings = embedder.encode(documents, convert_to_numpy=True)
    question_embedding = embedder.encode([question], convert_to_numpy=True)
    doc_norms = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    q_norm = question_embedding / np.linalg.norm(question_embedding, axis=1, keepdims=True)
    similarities = (doc_norms @ q_norm.T).flatten()
    ranked_indices = np.argsort(-similarities)[:top_k]
    return [(table_names[i], similarities[i]) for i in ranked_indices]


def get_linked_schema_text(db_path, question, embedder, top_k=5):
    schema = get_db_schema(db_path)
    table_names, documents = build_table_documents(schema)
    relevant = retrieve_relevant_tables(question, table_names, documents, embedder, top_k=top_k)
    relevant_table_names = [t[0] for t in relevant]
    return schema_to_text(schema, tables_filter=relevant_table_names), relevant
