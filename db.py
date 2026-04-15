"""
Database module for chat history management
Stores conversation messages in SQLite database
"""

import sqlite3
from datetime import datetime

DB_NAME = 'chat_history.db'

def init_db():
    """Initialize the database and create tables if they don't exist"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized: {DB_NAME}")

def save_message(role, content):
    """
    Save a message to the database
    
    Args:
        role (str): Either 'User' or 'Assistant'
        content (str): The message content
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute(
        "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
        (role, content, datetime.now())
    )
    
    conn.commit()
    conn.close()

def get_last_messages(limit=10):
    """
    Retrieve the last N messages from the database
    
    Args:
        limit (int): Number of messages to retrieve (default: 10)
    
    Returns:
        list: List of tuples (role, content) in chronological order
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute(
        "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    
    messages = c.fetchall()
    conn.close()
    
    # Reverse to get chronological order (oldest to newest)
    return list(reversed(messages))

def clear_history():
    """Clear all messages from the database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("DELETE FROM messages")
    
    conn.commit()
    conn.close()
    print("Chat history cleared")

def get_message_count():
    """Get total number of messages in the database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM messages")
    count = c.fetchone()[0]
    
    conn.close()
    return count

if __name__ == "__main__":
    # Test the database functions
    init_db()
    
    # Test saving messages
    save_message("User", "Hello, how are you?")
    save_message("Assistant", "I'm doing great! How can I help you today?")
    
    # Test retrieving messages
    messages = get_last_messages(limit=5)
    print("\nLast messages:")
    for role, content in messages:
        print(f"{role}: {content}")
    
    # Test message count
    count = get_message_count()
    print(f"\nTotal messages in database: {count}")