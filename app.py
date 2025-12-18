import streamlit as st
import os
import json
import pandas as pd
import requests

# --- CONFIG ---
st.set_page_config(page_title="AI Email Editor", page_icon="📧", layout="wide")
 
# --- UI HEADER ---
st.title("📧 AI Email Editing Tool")
st.header("Email Viewer")
st.write("Pick a dataset, then select an email ID to display.")

# --- DATASET SELECTION (Main area, no sidebar) ---
dataset_choice = st.selectbox(
    "Choose dataset",
    options=["shorten", "lengthen", "tone"],
    index=0,
)

def _read_jsonl(path: str) -> pd.DataFrame:
    """Read a JSONL file into a DataFrame with fallback parsing."""
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        # Fallback: manual line parsing
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pd.DataFrame.from_records(records)

@st.cache_data(show_spinner=False)
def load_dataset(choice: str) -> pd.DataFrame:
    path = os.path.join("datasets", f"{choice}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{choice}' not found at {path}")
    return _read_jsonl(path)

try:
    df = load_dataset(dataset_choice)
    if df.empty:
        st.warning("The selected dataset is empty.")
    else:
        # Determine ID column; default to 'id' or generate one
        id_col = "id" if "id" in df.columns else None
        if id_col is None:
            df = df.copy()
            df["_id"] = df.index + 1
            id_col = "_id"

        email_ids = df[id_col].tolist()
        selected_id = st.selectbox("Select Email ID", options=email_ids)

        row = df[df[id_col] == selected_id]
        if row.empty:
            st.error(f"No record found with ID {selected_id}.")
        else:
            rec = row.iloc[0]
            st.markdown(f"### ✉️ Email ID: `{selected_id}`")
            sender = rec.get("sender", None)
            subject = rec.get("subject", None)
            content = rec.get("content", None)

            if sender is not None:
                st.markdown(f"**From:** {sender}")
            if subject is not None:
                st.markdown(f"**Subject:** {subject}")

            if content is not None:
                st.text_area(
                    "Email Content",
                    value=str(content),
                    height=250,
                    key=f"email_text_{selected_id}",
                )
            else:
                st.subheader("Raw Record")
                st.json(rec.to_dict())

            st.divider()
            st.subheader("Actions")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.button("Elaborate", key=f"btn_elaborate_{selected_id}")
            with col2:
                st.button("Shorten", key=f"btn_shorten_{selected_id}")
            with col3:
                st.button("Change Tone", key=f"btn_change_tone_{selected_id}")
                tone_choice = st.selectbox(
                    "Tone",
                    options=["friendly", "sympathetic", "professional"],
                    index=0,
                    key=f"tone_select_{selected_id}",
                )
                

            with st.expander("Columns"):
                st.write(list(df.columns))
            st.caption(f"Rows: {len(df)} · Columns: {len(df.columns)}")
except FileNotFoundError as e:
    st.error(str(e))
except Exception as e:
    st.error(f"Failed to load dataset: {e}")