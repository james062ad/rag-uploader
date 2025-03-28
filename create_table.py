import psycopg2

# ---------------------------------------------
# ✅ Database connection configuration
# ---------------------------------------------
DB_NAME = "rag"
DB_USER = "postgres"
DB_PASSWORD = "postgres"
DB_HOST = "localhost"
DB_PORT = 5432

try:
    # ---------------------------------------------
    # ✅ Connect to the PostgreSQL + pgvector database
    # ---------------------------------------------
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    # ---------------------------------------------
    # ✅ Create a cursor to execute SQL commands
    # ---------------------------------------------
    cur = conn.cursor()

    # ---------------------------------------------
    # ✅ SQL to create the 'papers' table
    # ---------------------------------------------
    create_table_query = """
    CREATE TABLE IF NOT EXISTS papers (
        id SERIAL PRIMARY KEY,
        title TEXT,
        summary TEXT,
        chunk TEXT NOT NULL,
        embedding vector(1536)
    );
    """

    cur.execute(create_table_query)
    conn.commit()
    print("✅ Table 'papers' created successfully.")

except Exception as e:
    print("❌ Error creating table:", e)

finally:
    # ---------------------------------------------
    # ✅ Clean up: close cursor and connection
    # ---------------------------------------------
    if 'cur' in locals():
        cur.close()
    if 'conn' in locals():
        conn.close()
