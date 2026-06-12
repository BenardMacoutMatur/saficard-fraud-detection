import sqlite3
import hashlib
import random
from datetime import datetime, timedelta

DB_PATH = 'saficard.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

def generate_card_number():
    groups = [''.join([str(random.randint(0,9)) for _ in range(4)]) for _ in range(4)]
    return ' '.join(groups)

def generate_expiry():
    future = datetime.now() + timedelta(days=365*3)
    return future.strftime("%m/%y")

def clean_row(row):
    """Convert sqlite3.Row to clean dict with all string fields decoded."""
    if row is None:
        return None
    d = dict(row)
    for key, val in d.items():
        if isinstance(val, bytes):
            d[key] = val.decode('utf-8')
        elif val is None:
            d[key] = ''
    return d

def init_db():
    conn = get_connection()
    c    = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            name         TEXT NOT NULL,
            age          INTEGER NOT NULL,
            gender       TEXT NOT NULL,
            occupation   TEXT NOT NULL,
            county       TEXT NOT NULL,
            card_number  TEXT NOT NULL,
            card_type    TEXT NOT NULL,
            expiry       TEXT NOT NULL,
            credit_limit REAL NOT NULL,
            balance      REAL NOT NULL,
            pin_hash     TEXT NOT NULL,
            created_at   TEXT NOT NULL
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS transactions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            merchant   TEXT NOT NULL,
            category   TEXT NOT NULL,
            amount_kes REAL NOT NULL,
            fraud_prob REAL NOT NULL,
            is_fraud   INTEGER NOT NULL,
            status     TEXT NOT NULL,
            timestamp  TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()

# ============================================================
# USER OPERATIONS
# ============================================================
def create_user(name, age, gender, occupation, county,
                credit_limit, card_type, pin):
    conn        = get_connection()
    c           = conn.cursor()
    card_number = generate_card_number()
    expiry      = generate_expiry()

    c.execute('''
        INSERT INTO users
        (name, age, gender, occupation, county, card_number, card_type,
         expiry, credit_limit, balance, pin_hash, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(name), int(age), str(gender), str(occupation),
        str(county), str(card_number), str(card_type),
        str(expiry), float(credit_limit), float(credit_limit),
        hash_pin(str(pin)),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    user_id = c.lastrowid
    conn.close()
    return user_id, card_number, expiry

def get_user_by_name(name):
    conn = get_connection()
    c    = conn.cursor()
    c.execute('SELECT * FROM users WHERE name = ?', (str(name),))
    row  = c.fetchone()
    conn.close()
    return clean_row(row)

def verify_pin(name, pin):
    user = get_user_by_name(name)
    if user and user['pin_hash'] == hash_pin(str(pin)):
        return user
    return None

def update_balance(user_id, new_balance):
    conn = get_connection()
    c    = conn.cursor()
    c.execute('UPDATE users SET balance = ? WHERE id = ?',
              (float(new_balance), int(user_id)))
    conn.commit()
    conn.close()

def name_exists(name):
    return get_user_by_name(name) is not None

# ============================================================
# TRANSACTION OPERATIONS
# ============================================================
def save_transaction(user_id, merchant, category,
                     amount_kes, fraud_prob, is_fraud, status):
    conn = get_connection()
    c    = conn.cursor()
    c.execute('''
        INSERT INTO transactions
        (user_id, merchant, category, amount_kes, fraud_prob,
         is_fraud, status, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        int(user_id), str(merchant), str(category),
        float(amount_kes), float(fraud_prob),
        int(is_fraud), str(status),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()

def get_user_transactions(user_id, limit=10):
    conn = get_connection()
    c    = conn.cursor()
    c.execute('''
        SELECT * FROM transactions
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT ?
    ''', (int(user_id), int(limit)))
    rows = c.fetchall()
    conn.close()
    return [clean_row(row) for row in rows]

def get_user_stats(user_id):
    conn = get_connection()
    c    = conn.cursor()
    c.execute('''
        SELECT
            COUNT(*)                                         AS total,
            SUM(CASE WHEN is_fraud=0 THEN 1 ELSE 0 END)    AS approved,
            SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END)    AS blocked,
            SUM(CASE WHEN is_fraud=0 THEN amount_kes ELSE 0 END) AS total_spent
        FROM transactions WHERE user_id = ?
    ''', (int(user_id),))
    row = c.fetchone()
    conn.close()
    return clean_row(row)

def get_all_transactions():
    conn = get_connection()
    c    = conn.cursor()
    c.execute('''
        SELECT t.*, u.name as user_name, u.card_number
        FROM transactions t
        JOIN users u ON t.user_id = u.id
        ORDER BY t.timestamp DESC
        LIMIT 100
    ''')
    rows = c.fetchall()
    conn.close()
    return [clean_row(row) for row in rows]

def get_all_users():
    conn = get_connection()
    c    = conn.cursor()
    c.execute('''
        SELECT id, name, age, gender, occupation, county,
               card_number, card_type, expiry,
               credit_limit, balance, created_at
        FROM users
    ''')
    rows = c.fetchall()
    conn.close()
    return [clean_row(row) for row in rows]

def get_system_stats():
    conn = get_connection()
    c    = conn.cursor()
    c.execute('''
        SELECT
            COUNT(*)                                      AS total_txns,
            SUM(CASE WHEN is_fraud=0 THEN 1 ELSE 0 END) AS approved,
            SUM(CASE WHEN is_fraud=1 THEN 1 ELSE 0 END) AS blocked,
            SUM(amount_kes)                               AS total_volume
        FROM transactions
    ''')
    stats = clean_row(c.fetchone())
    c.execute('SELECT COUNT(*) AS total_users FROM users')
    stats['total_users'] = c.fetchone()['total_users']
    conn.close()
    return stats