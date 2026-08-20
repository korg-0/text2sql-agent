
import sqlite3
import pandas as pd


FEW_SHOT_EXAMPLES = """
Example 1:
Schema: CREATE TABLE employees (id INT PRIMARY KEY, name TEXT, department TEXT, salary INT)
Question: How many employees are there?
SQL: SELECT COUNT(*) FROM employees

Example 2:
Schema: CREATE TABLE orders (order_id INT PRIMARY KEY, customer_id INT, amount INT, order_date TEXT)
CREATE TABLE customers (customer_id INT PRIMARY KEY, name TEXT, city TEXT)
Question: What is the total order amount for each customer, showing customer name?
SQL: SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON c.customer_id = o.customer_id GROUP BY c.name

Example 3:
Schema: CREATE TABLE products (product_id INT PRIMARY KEY, name TEXT, price INT, category TEXT)
Question: Find products priced above the average price.
SQL: SELECT name FROM products WHERE price > (SELECT AVG(price) FROM products)
"""


def classify_difficulty(question, schema_text, client, model="openai/gpt-oss-120b"):
    prompt = f"""You are an expert SQL difficulty classifier. Given a database schema and a question, classify the difficulty of the SQL query needed as one of: EASY, NON-NESTED, NESTED.

EASY: single table, simple WHERE/COUNT/basic aggregation, no joins.
NON-NESTED: requires JOINs across multiple tables and/or GROUP BY/HAVING, but no subqueries.
NESTED: requires subqueries, set operations (INTERSECT/EXCEPT/UNION), or nested logic.

Schema:
{schema_text}

Question: {question}

Respond with ONLY one word: EASY, NON-NESTED, or NESTED."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().upper()


def generate_sql(question, schema_text, difficulty, client, model="openai/gpt-oss-120b"):
    if difficulty == "NESTED":
        reasoning_instruction = "This query likely requires a subquery or set operation. Think step by step about what intermediate result you need before writing the final SQL."
    elif difficulty == "NON-NESTED":
        reasoning_instruction = "This query likely requires JOINs and/or aggregation. Identify which tables need to be joined and on what keys."
    else:
        reasoning_instruction = "This is a simple query on a single table."

    prompt = f"""You are an expert SQL generator for SQLite databases. Generate a single, syntactically correct SQL query that answers the question.

{FEW_SHOT_EXAMPLES}

IMPORTANT RULES:
- Use exact equality (=) for text comparisons by default. Only use LIKE with wildcards if the question explicitly implies partial matching (e.g. "contains", "starts with", "includes the word").
- Match string literal casing to how it would realistically appear in the data (e.g. "Republic" not "republic") based on the sample rows shown in the schema.
- Prefer simple JOIN + GROUP BY + HAVING over subqueries when both achieve the same result, as it is more idiomatic SQL.

Now generate SQL for this case:

Schema:
{schema_text}

Question: {question}

Difficulty: {difficulty}
Hint: {reasoning_instruction}

Respond with ONLY the SQL query, no explanation, no markdown formatting, no backticks."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    sql = response.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def execute_sql(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return True, result
    except Exception as e:
        return False, str(e)


def generate_sql_with_correction(question, schema_text, difficulty, db_path, client,
                                   model="openai/gpt-oss-120b", max_retries=3):
    sql = generate_sql(question, schema_text, difficulty, client, model)

    for attempt in range(max_retries):
        success, result = execute_sql(sql, db_path)
        if success:
            return sql, result, attempt + 1
        else:
            correction_prompt = f"""The following SQL query failed when executed against the database.

Schema:
{schema_text}

Question: {question}

SQL that failed:
{sql}

Error:
{result}

Fix the SQL query. Respond with ONLY the corrected SQL query, no explanation, no markdown."""

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": correction_prompt}],
                temperature=0
            )
            sql = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").strip()

    success, result = execute_sql(sql, db_path)
    return sql, result, max_retries if success else -1
