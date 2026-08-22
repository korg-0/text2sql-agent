

import os
import sys
import sqlite3
import pandas as pd
import gradio as gr
import spaces
from groq import Groq
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/src')
from schema_linking import get_linked_schema_text
from sql_generation import classify_difficulty, generate_sql_with_correction

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "spider_databases", "validation_database")

AVAILABLE_DBS = sorted(os.listdir(DB_DIR)) if os.path.exists(DB_DIR) else []

EXAMPLE_QUESTIONS = {
    "concert_singer": "What is the average age of singers from France?",
    "world_1": "Which countries have a republic as their form of government?",
    "car_1": "How many car makers are there from each country?",
}


@spaces.GPU
def run_agent(db_id, question, top_k=6):
  
    if not question or not question.strip():
        return "", "", "", pd.DataFrame(), "Please enter a question."

    db_path = os.path.join(DB_DIR, db_id, f"{db_id}.sqlite")

    try:
        schema_text, relevant_tables = get_linked_schema_text(db_path, question, embedder, top_k=top_k)
        difficulty = classify_difficulty(question, schema_text, client)
        sql, result, attempts = generate_sql_with_correction(
            question, schema_text, difficulty, db_path, client
        )

        linked_tables_str = ", ".join([f"{t} (score: {s:.2f})" for t, s in relevant_tables])

        if isinstance(result, pd.DataFrame):
            return sql, difficulty, linked_tables_str, result, f"Success (attempts: {attempts})"
        else:
            return sql, difficulty, linked_tables_str, pd.DataFrame(), f"Execution failed: {result}"

    except Exception as e:
        return "", "", "", pd.DataFrame(), f"Error: {str(e)}"


def load_example(db_id):
    return EXAMPLE_QUESTIONS.get(db_id, "")

with gr.Blocks(title="Text-to-SQL Agent") as demo:
    gr.Markdown("""
    # 🔍 Enterprise Text-to-SQL Agent
    Schema-linking + decomposed prompting (DIN-SQL style) + execution self-correction.
    Ask a natural language question about a selected database and get executable SQL + results.

    *Built on the Spider benchmark. Backend: Groq (openai/gpt-oss-120b). ~91% execution accuracy on evaluated sample.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            db_dropdown = gr.Dropdown(choices=AVAILABLE_DBS, value=AVAILABLE_DBS[0] if AVAILABLE_DBS else None, label="Database")
            question_input = gr.Textbox(label="Your question", placeholder="e.g. How many singers do we have?", lines=2)
            submit_btn = gr.Button("Generate SQL", variant="primary")

            gr.Markdown("**Try an example:**")
            example_btn = gr.Button("Load example question for this DB")

        with gr.Column(scale=1):
            difficulty_output = gr.Textbox(label="Classified Difficulty")
            linked_tables_output = gr.Textbox(label="Linked Schema (Retrieved Tables)")
            status_output = gr.Textbox(label="Execution Status")

    sql_output = gr.Code(label="Generated SQL", language="sql")
    result_output = gr.Dataframe(label="Query Result", wrap=True, max_height=400)

    submit_btn.click(
        fn=run_agent,
        inputs=[db_dropdown, question_input],
        outputs=[sql_output, difficulty_output, linked_tables_output, result_output, status_output]
    )

    example_btn.click(
        fn=load_example,
        inputs=[db_dropdown],
        outputs=[question_input]
    )

if __name__ == "__main__":
    demo.launch()
