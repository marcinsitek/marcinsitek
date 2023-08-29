import sqlite3


def get_max_date(db: str, table: str, date_column: str):
    sql = f"""
        SELECT max({date_column}) FROM {table};
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    res = cur.execute(sql)
    value = res.fetchone()
    con.close()
    if value:
        return value[0]
    else:
        None


def truncate_table(db: str, table: str):
    con = sqlite3.connect(db)
    con.execute(f"DELETE FROM {table};")
    con.commit()
    con.close()