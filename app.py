"""Streamlit chat UI for Chat Over Tabular Data."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import run_pipeline

st.set_page_config(page_title="Chat Over Tabular Data", page_icon="📊", layout="centered")

st.title("📊 Chat Over Tabular Data")
st.caption("Ask natural language questions about your Clients, Invoices, and InvoiceLineItems.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("View SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("data") is not None and len(msg["data"]) > 0:
            with st.expander("View data"):
                st.dataframe(msg["data"], use_container_width=True)

if prompt := st.chat_input("Ask a question about the data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_pipeline(prompt)

        if result["error"]:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Error: {result['error']}",
                "sql": result["sql"],
            })
        else:
            st.markdown(result["answer"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sql": result["sql"],
                "data": result["data"],
            })
