#!/usr/bin/env python3
"""
Test script for vLLM OpenAI-compatible API.
Works with any model (text or multimodal).

Usage:
  python3 test_vllm.py <api_url>

Example:
  python3 test_vllm.py http://34.50.100.25:8000/v1
"""

import sys
import json
import time
import urllib.request
import urllib.error

def wait_for_api(base_url, timeout=600):
    """Wait for the vLLM API to be ready."""
    print(f"Waiting for vLLM API at {base_url} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=10) as resp:
                data = json.loads(resp.read())
                print(f"  API is up! Models: {[m['id'] for m in data.get('data', [])]}")
                return True
        except Exception:
            elapsed = int(time.time() - start)
            sys.stdout.write(f"\r  Waiting... {elapsed}s elapsed")
            sys.stdout.flush()
            time.sleep(5)
    print(f"\nTimeout after {timeout}s. vLLM may still be loading.")
    return False


def test_chat(base_url, model_name):
    """Test a chat completion request."""
    print(f"\n--- Chat Completion Test ---")

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Explain what Mixture-of-Experts is in one sentence."}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    elapsed = time.time() - start

    print(f"Response: {data['choices'][0]['message']['content']}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Tokens: {data['usage']['completion_tokens']}")
    print(f"Tokens/sec: {data['usage']['completion_tokens'] / elapsed:.1f}")


def test_streaming(base_url, model_name):
    """Test streaming chat completion."""
    print(f"\n--- Streaming Test ---")

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5."}
        ],
        "max_tokens": 50,
        "temperature": 0.3,
        "stream": True
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    print("Streaming response: ", end="", flush=True)
    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                delta = chunk["choices"][0]["delta"].get("content", "")
                print(delta, end="", flush=True)
    print()


def test_throughput(base_url, model_name):
    """Test concurrent requests for throughput."""
    import concurrent.futures

    print(f"\n--- Throughput Test (5 concurrent requests) ---")

    def single_request(idx):
        payload = json.dumps({
            "model": model_name,
            "messages": [
                {"role": "user", "content": f"Write a haiku about cloud computing. Variation {idx}."}
            ],
            "max_tokens": 80,
            "temperature": 0.8
        }).encode()

        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        start = time.time()
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        elapsed = time.time() - start
        return idx, data['usage']['completion_tokens'], elapsed

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(single_request, i) for i in range(5)]
        results = [f.result() for f in futures]
    total_time = time.time() - start

    total_tokens = sum(r[1] for r in results)
    print(f"  5 requests completed in {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Aggregate throughput: {total_tokens / total_time:.1f} tokens/sec")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_vllm.py <api_url>")
        print("Example: python3 test_vllm.py http://34.50.100.25:8000/v1")
        sys.exit(1)

    base_url = sys.argv[1].rstrip("/")

    # Wait for API
    if not wait_for_api(base_url):
        print("API not ready. Check: gcloud compute ssh vllm-server -- docker logs vllm")
        sys.exit(1)

    # Get model name from /models endpoint
    with urllib.request.urlopen(f"{base_url}/models", timeout=10) as resp:
        models = json.loads(resp.read())
    model_name = models["data"][0]["id"]
    print(f"Using model: {model_name}")

    # Run tests
    test_chat(base_url, model_name)
    test_streaming(base_url, model_name)
    test_throughput(base_url, model_name)

    print("\n✅ All tests passed!")
