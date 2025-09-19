import os
import json
import sqlite3
import pandas as pd
import streamlit as st
from openai import OpenAI
import getpass

# Initialize OpenAI client

# os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
# # client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# client = OpenAI()

# ----------------- API Key Handling -----------------
# if "openai_api_key" not in st.session_state:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
#     st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY")

# if not st.session_state.openai_api_key:
#     st.session_state.openai_api_key = st.text_input(
#         "Enter your OpenAI API key:", 
#         type="password"
#     )

# if not st.session_state.openai_api_key:
#     st.warning("OpenAI API key not found. Please provide to continue.")
#     st.stop()

open_ai_key = st.secrets["OPENAI_API_KEY"]["key"]
client = OpenAI(api_key=open_ai_key)

# ------------------ Natural Sentence Generators ------------------
def make_sentence_template(row):
    # base = f"On {row['Date']}, ${row['Amount']} was {row['Transaction_Type']}ed to account {row['Account_ID']}."

    base = f"On {row['Date']} at {row['Time']} with Transaction reference {row['Transaction_ID']}, a {row['Transaction_Type'].lower()} " \
           f"of {row['Amount']:,.2f} in currency {row['Currency']} was posted to {row['Account_Type']} with ID {row['Account_ID']} under entity name {row['Entity_Name']}, {row['Entity_Type']} {row['Entity_Segment']} with {row['This_ID_Type']} ID {row['This_ID']}. " \
           f"The transaction was made against {row['Contra_Pair_ID_Type']} ID {row['Contra_Pair_ID']}. " \
           f"The transaction was for {row['Event_Type']} for {row['Reference_Type']} reference {row['Reference']} and was made at {row['Bank_Name']} in {row['Bank_Town_Country']} and was flagged as {row['Status']}. "

    if pd.notna(row.get("Natural_Language_Sentence", "")) and row["Natural_Language_Sentence"]:
        base += f" ({row['Natural_Language_Sentence']})"
    return base

def make_sentences_gpt(df, batch_size=100, model="gpt-4o-mini"):
    sentences = {}
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start+batch_size]
        rows_text = "\n".join([f"{r.row_id}: {r.to_dict()}" for r in chunk.itertuples()])

        prompt = f"""Convert each transaction row into a natural language sentence optimized for AI-LLM processing and evaluation.
                    Return JSON list of objects with fields transaction_id, transaction_type and sentence.

                    Rows:
                    {rows_text}
                    """

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            data = json.loads(resp.choices[0].message.content)
            for item in data:
                sentences[item["Row_ID"]] = item["sentence"]
        except Exception as e:
            st.error(f"Error parsing GPT output: {e}")

    df["Natural_Language_Sentence"] = df["Row_ID"].map(sentences)
    return df

# ------------------ Database Setup ------------------
def init_db(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            Row_ID INTEGER PRIMARY KEY,
              Transaction_ID TEXT,
              Transaction_Type TEXT,
              Date TEXT,
              Time TEXT,
              Entity_Name TEXT,
              Entity_Type TEXT,
              Entity_Segment TEXT,
              This_ID_Type TEXT,
              This_ID TEXT,
              Contra_Pair_ID_Type TEXT,
              Contra_Pair_ID TEXT,
              Amount REAL,
              Currency TEXT,
              Account_Type TEXT,
              Account_ID TEXT,
              Event_Type TEXT,
              Reference_Type TEXT,
              Reference TEXT,
              Bank_Name TEXT,
              Bank_Town_Country TEXT,
              Status TEXT,
              NLS TEXT,
              Embedding BLOB
        )
    """)
    conn.commit()

def store_transactions(df, conn):
    c = conn.cursor()
    for row in df.itertuples(index=False):
        c.execute("""
            INSERT OR REPLACE INTO transactions
             (Transaction_ID,
              Transaction_Type,
              Date,
              Time,
              Entity_Name,
              Entity_Type,
              Entity_Segment,
              This_ID_Type,
              This_ID,
              Contra_Pair_ID_Type,
              Contra_Pair_ID,
              Amount,
              Currency,
              Account_Type,
              Account_ID,
              Event_Type,
              Reference_Type,
              Reference,
              Bank_Name,
              Bank_Town_Country,
              Status,
              NLS,
              Embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row.Transaction_ID,
            row.Transaction_Type,
            str(row.Date),
            str(row.Time),
            row.Entity_Name,
            row.Entity_Type,
            row.Entity_Segment,
            row.This_ID_Type,
            row.This_ID,
            row.Contra_Pair_ID_Type,
            row.Contra_Pair_ID,
            row.Amount,
            row.Currency,
            row.Account_Type,
            row.Account_ID,
            row.Event_Type,
            row.Reference_Type,
            row.Reference,
            row.Bank_Name,
            row.Bank_Town_Country,
            row.Status,
            row.Natural_Language_Sentence,
            None  # placeholder for embedding
        ))
    conn.commit()

# ------------------ Embedding Functions ------------------
def embed_text(text, model="text-embedding-3-small"):
    
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding

def create_embeddings(conn):
    c = conn.cursor()
    rows = c.execute("SELECT Row_ID, NLS FROM transactions WHERE embedding IS NULL").fetchall()
    for row_id, sentence in rows:
        vec = embed_text(sentence)
        c.execute("UPDATE transactions SET embedding = ? WHERE Row_ID = ?", (json.dumps(vec), row_id))
    conn.commit()

def search_similar(conn, query, top_k=5):
    query_vec = embed_text(query)
    c = conn.cursor()
    rows = c.execute("SELECT row_id, NLS, embedding FROM transactions").fetchall()

    # Compute cosine similarity manually
    import numpy as np
    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    scored = []
    for tid, sent, emb in rows:
        emb_vec = json.loads(emb)
        score = cosine(query_vec, emb_vec)
        scored.append((score, tid, sent))
    scored.sort(reverse=True)
    return scored[:top_k]

# ------------------ Query Answering ------------------
def answer_query(conn, query):
    
    similar = search_similar(conn, query)
    context = "\n".join([f"{tid}: {sent}" for _, tid, sent in similar])

    prompt = f"""You are a Chief Data Officer and an expert Banking Advisor and Assistant and a Senior Business and Data Analyst. Use the following transaction records to answer question as data held in a core banking system of the bank.
                In the context of evaluating the data, consider the fact that the transactions are in pairs of debit and credit entries. A pair is identified by a unique Transaction_ID.
                Consider also interest payment, penalty payment and charge payment as income or profit for the bank.
                The data will contain customer with Customer_ID. A customer may have multiple accounts with Account_ID in each. 
                If prompted with task to aggregate summary by customer, the related accounts will be aggregated. Amount can be aggregated into a given base currency of USD if not given.
Context:
{context}

Question: {query}
Answer:"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content

# ------------------ Streamlit App ------------------
st.title("💳 Banking Transactions AI Demo")

from chardet import detect

# uploaded = st.file_uploader("Upload your transactions CSV/XLSX", type=["csv","xlsx"])
uploaded = True

if uploaded:
#     with open(uploaded, 'rb') as f :
#         result = detect(f.read())
#         print(result['encoding']) # Detects encoding
    path = "./data/transactions.xlsx"        
    # df = pd.read_csv(path, encoding=result['encoding'])
    df = pd.read_excel(path)

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    gen_mode = st.radio(
        "How should natural sentences be generated?",
        ["Template (fast, free)", "GPT-4o-mini (higher quality, cost)"]
    )

    if st.button("Process & Store"):
        if gen_mode == "Template (fast, free)":
            df["Natural_Language_Sentence"] = df.apply(make_sentence_template, axis=1)
        else:
            df = make_sentences_gpt(df, batch_size=50)

        conn = sqlite3.connect("./data/transactions.db")
        init_db(conn)
        store_transactions(df, conn)
        create_embeddings(conn)
        conn.close()

        st.success("Transactions stored with embeddings!")

query = st.text_input("Ask a question about the transactions:")
if query:
    conn = sqlite3.connect("./data/transactions.db")
    answer = answer_query(conn, query)
    conn.close()

    st.subheader("Answer")
    st.write(answer)
