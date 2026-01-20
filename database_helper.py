import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager

# Pool
_pool = None

def init_pool():
    global _pool
    _pool = ThreadedConnectionPool(
        minconn=2,
        maxconn=20,
        host="localhost",
        port=5432,
        user="saurabhkamboj",
        database="wine_reviews"
    )

@contextmanager
def get_connection():
    conn = _pool.getconn()
    try:
        yield conn
    finally:
        _pool.putconn(conn)

# Search
def search_reviews(query_embedding=None, top_k=10, min_similarity=0.05, taster_name=None,
                   min_points=None, max_points=None, min_price=None, max_price=None):
    select_cols = (
        'id, title, variety, winery, country, province, description, '
        'points, price, taster_name, taster_twitter_handle'
    )

    conditions = []
    params = []

    if query_embedding is not None:
        conditions.append("1 - (embedding <=> %s::vector) > %s")
        params.extend([query_embedding, min_similarity])

    if taster_name is not None:
        conditions.append("LOWER(taster_name) = LOWER(%s)")
        params.append(taster_name)

    if min_points is not None:
        conditions.append("points >= %s")
        params.append(min_points)

    if max_points is not None:
        conditions.append("points <= %s")
        params.append(max_points)

    if min_price is not None:
        conditions.append("price >= %s")
        params.append(min_price)

    if max_price is not None:
        conditions.append("price <= %s")
        params.append(max_price)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    if query_embedding is not None:
        sql = f"""
            SELECT {select_cols}, 1 - (embedding <=> %s::vector) AS similarity
            FROM reviews
            WHERE {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [query_embedding] + params + [query_embedding, top_k]
    else:
        sql = f"""
            SELECT {select_cols}
            FROM reviews
            WHERE {where_clause}
            ORDER BY points DESC NULLS LAST, price NULLS LAST
            LIMIT %s
        """
        params.append(top_k)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        cur.close()

    results = []
    for row in rows:
        results.append({
            'id': row[0],
            'title': row[1],
            'variety': row[2],
            'winery': row[3],
            'country': row[4],
            'province': row[5],
            'description': row[6],
            'points': row[7],
            'price': float(row[8]) if row[8] is not None else None,
            'taster_name': row[9],
            'taster_twitter_handle': row[10],
            'similarity': float(row[11]) if query_embedding is not None else None
        })
    return results
