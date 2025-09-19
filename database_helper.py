import psycopg2

def get_conn():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        user="saurabhkamboj",
        database="wine_reviews"
    )

def search_reviews(query_embedding=None, top_k=10, min_similarity=0.05, taster_name=None,
                   min_points=None, max_points=None, min_price=None, max_price=None):
    select_cols = (
        'id, title, variety, winery, country, province, description, '
        'points, price, taster_name, taster_twitter_handle'
    )

    if query_embedding is not None:
        sql = f"""
            SELECT
                {select_cols},
                1 - (embedding <=> %s::vector) AS similarity
            FROM reviews
            WHERE (LOWER(taster_name) = LOWER(COALESCE(%s, taster_name)))
            AND (points >= COALESCE(%s, points))
            AND (points <= COALESCE(%s, points))
            AND (price  >= COALESCE(%s, price))
            AND (price  <= COALESCE(%s, price))
            AND 1 - (embedding <=> %s::vector) > %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = [
            query_embedding,
            taster_name,
            min_points,
            max_points,
            min_price,
            max_price,
            query_embedding, min_similarity,
            query_embedding, top_k
        ]
    else:
        sql = f"""
            SELECT
                {select_cols}
            FROM reviews
            WHERE (LOWER(taster_name) = LOWER(COALESCE(%s, taster_name)))
            AND (points >= COALESCE(%s, points))
            AND (points <= COALESCE(%s, points))
            AND (price  >= COALESCE(%s, price))
            AND (price  <= COALESCE(%s, price))
            ORDER BY points DESC NULLS LAST, price NULLS LAST
            LIMIT %s;
        """
        params = [
            taster_name,
            min_points,
            max_points,
            min_price,
            max_price,
            top_k
        ]

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
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
