import sqlite3
import datetime

def initialize_database(dbName: str):
    conn = sqlite3.connect(f"path/to/dbs/{dbName}.db")
    c = conn.cursor()

    c.execute("DROP TABLE IF EXISTS vehicle_counts")

    query = '''CREATE TABLE IF NOT EXISTS vehicle_counts 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  timeStamp TEXT,
                  car INTEGER,
                  motorcycle INTEGER,
                  van INTEGER,
                  rickshaw INTEGER,
                  bus INTEGER,  
                  truck INTEGER    
                  );'''

    c.execute(query)

    conn.commit()
    conn.close()

def add_reading(dbName: str, reading: dict):
    conn = sqlite3.connect(f"path/to/dbs/{dbName}.db")
    c = conn.cursor()

    query = '''INSERT INTO vehicle_counts (date, timeStamp, car, motorcycle, van, rickshaw, bus, truck)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?);'''
    c.execute(query, (
        reading["date"],
        reading["timeStamp"],
        reading["car"],
        reading["motorcycle"],
        reading["van"],
        reading["rickshaw"],
        reading["bus"],
        reading["truck"]
    ))
    conn.commit()
    conn.close()