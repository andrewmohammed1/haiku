import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
import json
import os
import asyncio
import random
from datetime import datetime
from typing import List, Dict, Any

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Talent Intelligence Suite", layout="wide")
st.title("🚀 Talent Intelligence (High-Efficiency Async Edition)")

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

# --- 3. ASYNC RATE LIMITER ---

class RateLimiter:
    """Concurrent-friendly token bucket rate limiter."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.tokens = requests_per_minute
        self.max_tokens = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.last_refill = None
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Acquire a token, waiting if necessary."""
        async with self.lock:
            now = asyncio.get_event_loop().time()

            # Initialize on first call
            if self.last_refill is None:
                self.last_refill = now

            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

            # If no tokens available, wait until one is available
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
                self.tokens = 1

            # Consume a token
            self.tokens -= 1

# --- 4. ASYNC PROCESSING ENGINE ---

async def process_batch_async(
    batch: pd.DataFrame,
    client: genai.Client,
    rate_limiter: RateLimiter,
    model: str = "gemini-3-flash-preview",
    max_retries: int = 5
) -> List[Dict[str, Any]]:
    """Process a single batch asynchronously with retry logic."""

    # Prepare batch payload
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

    # Retry loop with exponential backoff
    for retry in range(max_retries):
        try:
            # Wait for rate limiter
            await rate_limiter.acquire()

            # Make API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={'response_mime_type': 'application/json'}
                )
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

            return results_list

        except Exception as e:
            if retry < max_retries - 1:
                # Exponential backoff with jitter
                wait_time = (2 ** retry) * 5 + random.uniform(2, 5)
                await asyncio.sleep(wait_time)
            else:
                # Final retry failed - return empty to avoid crashing
                st.warning(f"⚠️ Batch failed after {max_retries} retries: {str(e)[:100]}")
                return []

    return []

async def process_all_batches(
    batches: List[pd.DataFrame],
    client: genai.Client,
    rate_limiter: RateLimiter,
    max_concurrent: int,
    progress_bar,
    status_msg,
    model: str = "gemini-3-flash-preview"
):
    """Process all batches with controlled concurrency."""

    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    total = len(batches)

    async def process_with_semaphore(batch_idx: int, batch: pd.DataFrame):
        nonlocal completed
        async with semaphore:
            results = await process_batch_async(batch, client, rate_limiter, model)

            # Save results immediately
            if results:
                save_to_cache(pd.DataFrame(results))

            # Update progress
            completed += 1
            progress_bar.progress(completed / total)
            status_msg.info(f"Processed {completed} / {total} batches")

            return results

    # Process all batches concurrently (but respecting semaphore limit)
    tasks = [process_with_semaphore(i, batch) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

# --- 5. UI SIDEBAR ---

with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password")

    # Batch size: how many people per request
    batch_size = st.slider("Batch Size (People per request)", 5, 20, 10)

    # Concurrent requests: how many batches to process simultaneously
    max_concurrent = st.slider("Max Concurrent Requests", 1, 10, 5,
                               help="Higher = faster, but may hit rate limits")

    # Rate limit (requests per minute)
    rpm_limit = st.slider("Rate Limit (RPM)", 10, 60, 15,
                          help="Personal tier: 15 RPM, Paid tier: 60+ RPM")

    if st.button("🗑️ Reset All Progress"):
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            st.rerun()

# --- 6. MAIN PROCESSING ---

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
            if st.button("🧠 Start Skills Mining (Async)"):
                # Initialize client and rate limiter
                client = genai.Client(api_key=api_key)
                rate_limiter = RateLimiter(requests_per_minute=rpm_limit)

                # Create batches
                batches = [to_process.iloc[i:i + batch_size]
                          for i in range(0, len(to_process), batch_size)]

                st.info(f"⚡ Processing {len(batches)} batches with up to {max_concurrent} concurrent requests")

                progress_bar = st.progress(0)
                status_msg = st.empty()

                # Run async processing
                start_time = datetime.now()
                asyncio.run(process_all_batches(
                    batches,
                    client,
                    rate_limiter,
                    max_concurrent,
                    progress_bar,
                    status_msg
                ))

                elapsed = (datetime.now() - start_time).total_seconds()
                st.success(f"✅ Done! Processed {len(batches)} batches in {elapsed:.1f}s ({len(to_process)} records)")
                st.rerun()
        elif not api_key:
            st.warning("Please enter an API Key.")

# --- 7. DASHBOARD ---
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
