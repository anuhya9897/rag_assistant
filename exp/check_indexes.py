import duckdb

con = duckdb.connect("kb.duckdb")
rows = con.execute(
    "SELECT index_name, table_name FROM duckdb_indexes() WHERE table_name='kb_chunks_local'"
).fetchall()
print(rows)
