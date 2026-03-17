import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import json
import os
import time
import random

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Talent Intelligence Suite", layout="wide")
st.title("🚀 Talent Intelligence (High-Efficiency Edition)")

# Local cache to prevent redundant API calls and save quota
CACHE_FILE = "processing_cache.csv"

# --- 2. DATA UTILITIES ---

def save_to_cache(new_data_df):
    """Saves results while preventing duplicate skill entries for the same ID."""
    if not os.path.isfile(CACHE_FILE):
        new_data_df.to_csv(CACHE_FILE, index=False)
    else:
        existing = pd.read_csv(CACHE_FILE)
        # QA Check: Avoid appending duplicates if a batch is partially retried
        combined = pd.concat([existing, new_data_df]).drop_duplicates(subset=["Worker ID", "Skill"])
        combined.to_csv(CACHE_FILE, index=False)

def load_cache():
    """Reads the current progress from the CSV file."""
    if os.path.isfile(CACHE_FILE):
        return pd.read_csv(CACHE_FILE, low_memory=False)
    return pd.DataFrame()

def clean_and_map_columns(df):
    """Standardizes column names from various HRIS export formats."""
    df.columns = [str(c).strip() for c in df.columns]
    mapping = {
        'Worker ID': ['workerid', 'employeeid', 'id', 'wid', 'personid'],
        'Worker Name': ['workername', 'employeename', 'name', 'fullname'],
        'Team': ['team', 'department', 'dept', 'org', 'organization'],
        'Job Title': ['jobtitle', 'position', 'role', 'title', 'jobprofile'],
        'Job Description': ['jobdescription', 'jd', 'summary', 'description']
    }
    
    new_columns = {}
    for col in df.columns:
        clean_col = str(col).lower().replace(" ", "").replace("_", "").replace("-", "")
        for standard_name, alternatives in mapping.items():
            if clean_col in alternatives:
                new_columns[col] = standard_name
                break
    
    df = df.rename(columns=new_columns)
    # If no description exists, use Title to give the AI at least some context
    if 'Job Title' in df.columns and 'Job Description' not in df.columns:
        df['Job Description'] = df['Job Title']
    return df

# --- 3. UI SIDEBAR ---

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password")
    
    # Recommendation: 10-15 for Personal Tier, 20+ for Paid Tier
    batch_size = st.slider("Batch Size (People per request)", 5, 20, 10)
    
    if st.button("🗑️ Reset All Progress"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.rerun()

# --- 4. PROCESSING ENGINE ---

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    df_clean = clean_and_map_columns(df_raw)
    
    if 'Worker ID' in df_clean.columns and 'Job Title' in df_clean.columns:
        # Load cache to determine who still needs processing
        processed_cache = load_cache()
        processed_ids = set(processed_cache['Worker ID'].astype(str).unique()) if not processed_cache.empty else set()
        to_process = df_clean[~df_clean['Worker ID'].astype(str).isin(processed_ids)]
        
        st.info(f"📁 Records: {len(df_clean)} | To Process: {len(to_process)}")
        
        if not to_process.empty and api_key:
            if st.button("🧠 Start Skills Mining"):
                # Initializing Client with your specified configuration
                client = genai.Client(api_key=api_key)
                
                progress_bar = st.progress(0)
                status_msg = st.empty()
                
                for i in range(0, len(to_process), batch_size):
                    batch = to_process.iloc[i : i + batch_size]
                    
                    # Truncate descriptions to 400 chars to save Tokens Per Minute (TPM)
                    batch_payload = []
                    for _, row in batch.iterrows():
                        batch_payload.append({
                            "id": str(row.get('Worker ID')),
                            "title": str(row.get('Job Title')),
                            "context": str(row.get('Job Description', ''))[:400]
                        })

                    prompt = f"""
                    Return a JSON array of skill objects for these IDs. 
                    Structure: [{{"id": "...", "skills": [{{"skill": "...", "category": "...", "level": "..."}}]}}]
                    DATA: {json.dumps(batch_payload)}
                    """

                    # --- RESILIENCE LOOP ---
                    success = False
                    retries = 0
                    while not success and retries < 5:
                        try:
                            # Using your specific model choice
                            response = client.models.generate_content(
                                model="gemini-3-flash-preview", 
                                contents=prompt,
                                config={'response_mime_type': 'application/json'}
                            )
                            
                            raw_results = json.loads(response.text)
                            results_list = []
                            
                            for emp in raw_results:
                                emp_id = str(emp.get('id'))
                                match = batch[batch['Worker ID'].astype(str) == emp_id]
                                if not match.empty:
                                    orig = match.iloc[0]
                                    for s in emp.get('skills', []):
                                        results_list.append({
                                            "Worker ID": emp_id,
                                            "Name": orig.get('Worker Name', 'N/A'),
                                            "Team": orig.get('Team', 'N/A'),
                                            "Job Title": orig.get('Job Title', 'N/A'),
                                            "Skill": s.get('skill', 'Unknown'),
                                            "Category": s.get('category', 'General'),
                                            "Proficiency": s.get('level', 'Foundational')
                                        })
                            
                            if results_list:
                                save_to_cache(pd.DataFrame(results_list))
                            
                            success = True
                            # GENEROUS BUFFER: 8 seconds prevents hitting the 15 RPM limit on Personal Tier
                            time.sleep(8)

                        except Exception as e:
                            retries += 1
                            # Exponential backoff: 10s, 20s, 40s...
                            wait = (2 ** retries) * 5 + random.uniform(2, 5)
                            status_msg.warning(f"⚠️ Rate limited. Waiting {int(wait)}s... (Attempt {retries}/5)")
                            time.sleep(wait)

                    # Update progress UI
                    progress_pct = min((i + batch_size) / len(to_process), 1.0)
                    progress_bar.progress(progress_pct)
                    status_msg.info(f"Processed {min(i + batch_size, len(to_process))} / {len(to_process)}")

                st.success("✅ Done!")
                st.rerun()
        elif not api_key:
            st.warning("Please enter an API Key.")

# --- 5. DASHBOARD ---
data = load_cache()
if not data.empty:
    st.divider()
    t1, t2, t3 = st.tabs(["📊 Insights", "🔍 Search", "📥 Export"])
    with t1:
        fig = px.treemap(data, path=['Team', 'Job Title', 'Skill'], color='Proficiency')
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        search = st.text_input("Filter by Skill or Name")
        mask = data.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
        st.dataframe(data[mask] if search else data)
    with t3:
        st.download_button("Download Full CSV", data.to_csv(index=False), "skills_inventory.csv")


