import os
import sqlite3
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

conn = sqlite3.connect("SalesDB/sales.db")

cursor = conn.cursor()

cursor.execute("SELECT * from orders")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
