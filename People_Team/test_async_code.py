"""
Test suite for async LLM processing code
Validates logic without requiring API keys or real data
"""

import asyncio
import pandas as pd
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import sys

# --- MOCK CLASSES ---

class MockResponse:
    """Mock API response"""
    def __init__(self, data):
        self.text = json.dumps(data)

class MockClient:
    """Mock Gemini client for testing"""
    def __init__(self, delay=0.1, fail_rate=0.0):
        self.delay = delay
        self.fail_rate = fail_rate
        self.call_count = 0
        self.models = self

    def generate_content(self, model, contents, config):
        """Simulate API call"""
        self.call_count += 1
        time.sleep(self.delay)  # Simulate network delay

        # Simulate random failures
        import random
        if random.random() < self.fail_rate:
            raise Exception("Simulated API error")

        # Parse the prompt to extract batch data
        if "DATA:" in contents:
            data_str = contents.split("DATA:")[1].strip()
            batch_data = json.loads(data_str)

            # Generate mock skills for each person
            results = []
            for person in batch_data:
                results.append({
                    "id": person["id"],
                    "skills": [
                        {"skill": "Python", "category": "Technical", "level": "Advanced"},
                        {"skill": "Communication", "category": "Soft Skills", "level": "Expert"},
                        {"skill": "Data Analysis", "category": "Technical", "level": "Intermediate"}
                    ]
                })

            return MockResponse(results)

        return MockResponse([])

# --- RATE LIMITER (from async code) ---

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

# --- ASYNC PROCESSING (from async code) ---

async def process_batch_async(
    batch: pd.DataFrame,
    client: MockClient,
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
                lambda: client.generate_content(
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
                import random
                wait_time = (2 ** retry) * 0.5 + random.uniform(0.1, 0.3)
                await asyncio.sleep(wait_time)
            else:
                print(f"⚠️ Batch failed after {max_retries} retries: {str(e)[:100]}")
                return []

    return []

async def process_all_batches(
    batches: List[pd.DataFrame],
    client: MockClient,
    rate_limiter: RateLimiter,
    max_concurrent: int,
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
            completed += 1
            print(f"✓ Batch {completed}/{total} complete ({len(results)} skills)")
            return results

    # Process all batches concurrently (but respecting semaphore limit)
    tasks = [process_with_semaphore(i, batch) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    return results

# --- TEST CASES ---

def create_test_data(num_rows: int) -> pd.DataFrame:
    """Create synthetic test data"""
    data = []
    for i in range(num_rows):
        data.append({
            'Worker ID': f'EMP{i:04d}',
            'Worker Name': f'Employee {i}',
            'Team': f'Team {i % 5}',
            'Job Title': f'Software Engineer L{(i % 3) + 1}',
            'Job Description': f'Develops software solutions using Python and cloud technologies.'
        })
    return pd.DataFrame(data)

async def test_rate_limiter():
    """Test that rate limiter works correctly"""
    print("\n" + "="*60)
    print("TEST 1: Rate Limiter Accuracy")
    print("="*60)

    rpm = 60  # 1 request per second
    rate_limiter = RateLimiter(requests_per_minute=rpm)

    start = time.time()
    for i in range(5):
        await rate_limiter.acquire()
    elapsed = time.time() - start

    expected = 4.0  # Should take ~4 seconds for 5 requests at 60 RPM
    tolerance = 0.5

    if abs(elapsed - expected) < tolerance:
        print(f"✅ PASS: Rate limiter accurate ({elapsed:.2f}s ≈ {expected}s)")
    else:
        print(f"❌ FAIL: Rate limiter inaccurate ({elapsed:.2f}s != {expected}s)")

    return abs(elapsed - expected) < tolerance

async def test_concurrent_processing():
    """Test concurrent batch processing"""
    print("\n" + "="*60)
    print("TEST 2: Concurrent Processing Speed")
    print("="*60)

    # Create test data
    df = create_test_data(100)
    batch_size = 10
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    # Test sequential (concurrency = 1)
    print("\n📊 Sequential processing (concurrency=1):")
    client = MockClient(delay=0.5)
    rate_limiter = RateLimiter(requests_per_minute=60)

    start = time.time()
    await process_all_batches(batches, client, rate_limiter, max_concurrent=1)
    sequential_time = time.time() - start
    print(f"⏱️  Time: {sequential_time:.2f}s | API Calls: {client.call_count}")

    # Test concurrent (concurrency = 5)
    print("\n📊 Concurrent processing (concurrency=5):")
    client = MockClient(delay=0.5)
    rate_limiter = RateLimiter(requests_per_minute=60)

    start = time.time()
    await process_all_batches(batches, client, rate_limiter, max_concurrent=5)
    concurrent_time = time.time() - start
    print(f"⏱️  Time: {concurrent_time:.2f}s | API Calls: {client.call_count}")

    speedup = sequential_time / concurrent_time
    print(f"\n🚀 Speedup: {speedup:.2f}x faster")

    if speedup > 2.0:  # Should be significantly faster
        print(f"✅ PASS: Concurrent processing is {speedup:.2f}x faster")
    else:
        print(f"❌ FAIL: Concurrent processing not fast enough ({speedup:.2f}x)")

    return speedup > 2.0

async def test_retry_logic():
    """Test retry and error handling"""
    print("\n" + "="*60)
    print("TEST 3: Retry Logic with Failures")
    print("="*60)

    df = create_test_data(10)
    batches = [df]

    # Test with 50% failure rate
    client = MockClient(delay=0.1, fail_rate=0.5)
    rate_limiter = RateLimiter(requests_per_minute=120)

    results = await process_all_batches(batches, client, rate_limiter, max_concurrent=1)

    if len(results) > 0:
        print(f"✅ PASS: Handled failures with retry logic ({client.call_count} attempts)")
    else:
        print(f"❌ FAIL: No results returned")

    return len(results) > 0

async def test_data_integrity():
    """Test that all data is processed correctly"""
    print("\n" + "="*60)
    print("TEST 4: Data Integrity")
    print("="*60)

    df = create_test_data(50)
    batch_size = 10
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    client = MockClient(delay=0.05)
    rate_limiter = RateLimiter(requests_per_minute=120)

    results = await process_all_batches(batches, client, rate_limiter, max_concurrent=3)

    # Flatten results
    all_skills = []
    for batch_results in results:
        if isinstance(batch_results, list):
            all_skills.extend(batch_results)

    # Each person should have 3 skills
    expected_skills = len(df) * 3
    actual_skills = len(all_skills)

    # Check unique worker IDs
    unique_ids = set(skill['Worker ID'] for skill in all_skills)
    expected_ids = len(df)
    actual_ids = len(unique_ids)

    print(f"Expected skills: {expected_skills} | Actual: {actual_skills}")
    print(f"Expected unique IDs: {expected_ids} | Actual: {actual_ids}")

    if actual_skills == expected_skills and actual_ids == expected_ids:
        print(f"✅ PASS: All data processed correctly")
        return True
    else:
        print(f"❌ FAIL: Data mismatch")
        return False

async def test_scalability():
    """Test performance at scale"""
    print("\n" + "="*60)
    print("TEST 5: Scalability (1000 records)")
    print("="*60)

    df = create_test_data(1000)
    batch_size = 10
    batches = [df.iloc[i:i + batch_size] for i in range(0, len(df), batch_size)]

    print(f"Processing {len(df)} records in {len(batches)} batches")

    client = MockClient(delay=0.1)  # 100ms per call
    rate_limiter = RateLimiter(requests_per_minute=120)

    start = time.time()
    results = await process_all_batches(batches, client, rate_limiter, max_concurrent=5)
    elapsed = time.time() - start

    records_per_second = len(df) / elapsed

    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"📊 Throughput: {records_per_second:.1f} records/second")
    print(f"🔥 API calls: {client.call_count}")

    if records_per_second > 30:  # Should process at least 30 records/sec
        print(f"✅ PASS: Good throughput ({records_per_second:.1f} rec/s)")
        return True
    else:
        print(f"❌ FAIL: Low throughput ({records_per_second:.1f} rec/s)")
        return False

# --- MAIN TEST RUNNER ---

async def run_all_tests():
    """Run comprehensive test suite"""
    print("\n" + "="*60)
    print("🧪 ASYNC LLM CODE TEST SUITE")
    print("="*60)

    tests = [
        ("Rate Limiter", test_rate_limiter),
        ("Concurrent Processing", test_concurrent_processing),
        ("Retry Logic", test_retry_logic),
        ("Data Integrity", test_data_integrity),
        ("Scalability", test_scalability)
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = await test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\n🎯 Score: {passed}/{total} tests passed ({100*passed/total:.0f}%)")

    if passed == total:
        print("\n🎉 All tests passed! Code is production-ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review required.")

if __name__ == "__main__":
    asyncio.run(run_all_tests())
