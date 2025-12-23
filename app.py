"""
AI Email Editing Tool

A Streamlit application that allows users to:
1. Browse email datasets (shorten, lengthen, tone)
2. Select individual emails by ID
3. Generate variations of emails using Azure OpenAI's GPT models
4. View and compare AI-generated email versions

The app uses Azure OpenAI ChatCompletions API to perform three main operations:
- Shorten: Condense emails while preserving key information
- Lengthen/Elaborate: Expand emails with additional context and details
- Change Tone: Rewrite emails in different tones (friendly, sympathetic, professional)

Configuration:
    - Datasets: JSONL files in ./datasets/ directory
    - Models: Deployed via Azure OpenAI (configured in generate.py)
    - Prompts: Defined in prompts.yaml
"""

import streamlit as st
import os
import json
import pandas as pd
import requests
from generate import GenerateEmail


def _read_jsonl(path: str) -> pd.DataFrame:
    """Read a JSONL file into a pandas DataFrame with fallback parsing.
    
    Attempts to read the JSONL file using pandas.read_json(). If that fails,
    manually parses each line as JSON and creates a DataFrame from records.
    
    Args:
        path (str): File path to the JSONL file.
    
    Returns:
        pd.DataFrame: DataFrame containing parsed JSONL records.
    
    Raises:
        JSONDecodeError: If a line cannot be parsed as JSON (gracefully skipped).
    """
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        # Fallback: manual line-by-line parsing for robustness
        records = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines gracefully
                    continue
        return pd.DataFrame.from_records(records)


def load_dataset(choice: str) -> pd.DataFrame:
    """Load a dataset from the datasets folder.
    
    Reads a JSONL file corresponding to the user's dataset choice.
    
    Args:
        choice (str): Dataset name ('shorten', 'lengthen', or 'tone').
    
    Returns:
        pd.DataFrame: DataFrame containing the dataset records.
    
    Raises:
        FileNotFoundError: If the dataset file does not exist.
    """
    path = os.path.join("datasets", f"{choice}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{choice}' not found at {path}")
    return _read_jsonl(path)


def generate_and_display(action: str, label: str, content: str, 
                        tone: str = None, selected_id: int = None, model: str = "gpt-4.1") -> None:
    """Generate an AI-edited email and display it in the UI.
    
    Calls the GenerateEmail API with the specified action, displays a loading
    spinner, and renders the result in a text area. Shows an error message
    if the API call fails.
    
    Args:
        action (str): Type of generation action ('shorten', 'lengthen', 'tone').
        label (str): Display label for the result section (e.g., 'Shortened', 'Friendly Tone').
        content (str): The original email content to be processed.
        tone (str, optional): Tone for the 'tone' action ('friendly', 'sympathetic', 'professional').
        selected_id (int, optional): Email ID for generating unique keys.
        model (str, optional): Azure deployment model name. Defaults to 'gpt-4.1'.
    
    Returns:
        None: Displays result in Streamlit UI.
    """
    spinner_msg = f"Generating {label.lower()} version..."
    generator = GenerateEmail(model=model)
    judge_generator = GenerateEmail(model="gpt-4.1")  # Judges always use gpt-4.1
    
    with st.spinner(spinner_msg):
        result = generator.generate(action, str(content), tone=tone)
        if result:
            st.divider()
            st.subheader(f"Generated Email ({label})")
            st.text_area(
                "Result",
                value=result,
                height=250,
                key=f"result_{action}_{selected_id}_{tone or ''}",
                disabled=False,  # Allow users to copy and edit results
            )

            # Judge the model's response on multiple metrics
            st.caption("Quality Ratings (Judge Models)")
            cols = st.columns(3)
            metrics = ["faithfulness", "completeness", "conciseness"]
            for idx, metric in enumerate(metrics):
                with cols[idx]:
                    judge_result = judge_generator.judge(metric, str(content), result)
                    if isinstance(judge_result, dict):
                        rating = judge_result.get("rating", "?")
                        explanation = judge_result.get("explanation", "")
                        st.markdown(f"**{metric.title()}**: {rating}")
                        st.caption(explanation)
                    else:
                        st.markdown(f"**{metric.title()}**")
                        st.caption(judge_result if judge_result else "No result")
        else:
            st.error("Failed to generate. Check API connection and credentials.")


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(page_title="AI Email Editor", page_icon="📧", layout="wide")

# =============================================================================
# PAGE HEADER
# =============================================================================

st.title("📧 AI Email Editing Tool")
st.header("Email Viewer")
st.write("Pick a dataset, then select an email ID to display and edit using AI.")

# =============================================================================
# DATASET SELECTION
# =============================================================================
# Allow users to toggle between three datasets: shorten, lengthen, tone

dataset_choice = st.selectbox(
    "Choose dataset",
    options=["shorten", "lengthen", "tone"],
    index=0,
    help="Select which dataset to browse. Each dataset contains email examples for different use cases.",
)

# =============================================================================
# MODEL SELECTION
# =============================================================================
# Allow users to toggle between available AI models

model_choice = st.selectbox(
    "Choose model",
    options=["gpt-4.1", "gpt-4o-mini"],
    index=0,
    help="Select which Azure OpenAI model to use for generation. gpt-4.1 is more powerful, gpt-4o-mini is faster.",
)


# =============================================================================
# LOAD AND DISPLAY DATASET
# =============================================================================

try:
    df = load_dataset(dataset_choice)
    
    if df.empty:
        st.warning("The selected dataset is empty.")
    else:
        # Determine ID column; create one if not present
        id_col = "id" if "id" in df.columns else None
        if id_col is None:
            df = df.copy()
            df["_id"] = df.index + 1
            id_col = "_id"

        email_ids = df[id_col].tolist()
        
        # =================================================================
        # EMAIL SELECTION
        # =================================================================
        # Use number input with arrows for intuitive navigation
        
        email_index = st.number_input(
            "Select Email ID (use arrows to navigate)",
            min_value=1,
            max_value=len(email_ids),
            value=1,
            step=1,
            key=f"email_index_{dataset_choice}",
            help="Choose an email by ID number. Use the up/down arrows to navigate.",
        )
        selected_id = email_ids[email_index - 1]  # Convert 1-indexed UI to 0-indexed list
        st.caption(f"Email ID: {selected_id}")

        row = df[df[id_col] == selected_id]
        if row.empty:
            st.error(f"No record found with ID {selected_id}.")
        else:
            rec = row.iloc[0]
            
            # =============================================================
            # DISPLAY SELECTED EMAIL
            # =============================================================
            
            st.markdown(f"### ✉️ Email ID: `{selected_id}`")
            
            # Display sender if available
            sender = rec.get("sender", None)
            if sender is not None:
                st.markdown(f"**From:** {sender}")
            
            # Display subject if available
            subject = rec.get("subject", None)
            if subject is not None:
                st.markdown(f"**Subject:** {subject}")

            # Display content in editable text area
            content = rec.get("content", None)
            if content is not None:
                st.text_area(
                    "Email Content",
                    value=str(content),
                    height=250,
                    key=f"email_text_{dataset_choice}_{selected_id}",
                )
            else:
                # Fallback: show raw record as JSON
                st.subheader("Raw Record")
                st.json(rec.to_dict())

            # =============================================================
            # ACTION BUTTONS
            # =============================================================
            # Three main actions: Elaborate, Shorten, Change Tone
            # Each triggers an API call to the LLM
            
            st.divider()
            st.subheader("✨ AI Actions")
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                elaborate_clicked = st.button(
                    "Elaborate",
                    key=f"btn_elaborate_{selected_id}",
                    help="Generate a longer, more detailed version of this email.",
                )
            with col2:
                shorten_clicked = st.button(
                    "Shorten",
                    key=f"btn_shorten_{selected_id}",
                    help="Generate a concise, shortened version of this email.",
                )
            with col3:
                change_tone_clicked = st.button(
                    "Change Tone",
                    key=f"btn_change_tone_{selected_id}",
                    help="Rewrite this email in a different tone.",
                )
                tone_choice = st.selectbox(
                    "Tone",
                    options=["friendly", "sympathetic", "professional"],
                    index=0,
                    key=f"tone_select_{selected_id}",
                    help="Select the desired tone for the rewritten email.",
                )

            # =============================================================
            # HANDLE BUTTON CLICKS & DISPLAY RESULTS
            # =============================================================
            # When a button is clicked, call the LLM and display the result
            
            if content is not None:
                if elaborate_clicked:
                    generate_and_display("lengthen", "Elaborate", content, selected_id=selected_id, model=model_choice)
                
                if shorten_clicked:
                    generate_and_display("shorten", "Shortened", content, selected_id=selected_id, model=model_choice)
                
                if change_tone_clicked:
                    generate_and_display(
                        "tone",
                        f"{tone_choice.title()} Tone",
                        content,
                        tone=tone_choice,
                        selected_id=selected_id,
                        model=model_choice,
                    )

            # =============================================================
            # DATASET METADATA
            # =============================================================
            
            with st.expander("📊 Dataset Info"):
                st.write(f"**Columns:** {', '.join(list(df.columns))}")
                st.write(f"**Total Rows:** {len(df)}")

except FileNotFoundError as e:
    st.error(f"Dataset Error: {str(e)}")
except Exception as e:
    st.error(f"Unexpected Error: Failed to load dataset - {str(e)}")
