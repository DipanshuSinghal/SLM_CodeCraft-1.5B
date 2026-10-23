import time
import requests

API_URL = "http://localhost:8000/v1/generate"

TEST_PROMPTS = [
    "Write a Python function named `add` that returns the sum of two numbers.",
    "Write a Python function named `is_prime` that checks if a given integer n is prime.",
    "Write a Python function named `reverse_string` that returns a reversed string."
]

def run_benchmarks():
    print("Starting API Evaluation Benchmark...\n" + "="*50)
    for idx, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"Test #{idx}: {prompt}")
        payload = {"prompt": prompt, "max_tokens": 128, "temperature": 0.1}
        
        try:
            start = time.time()
            res = requests.post(API_URL, json=payload)
            res_data = res.json()
            duration = round((time.time() - start) * 1000, 2)
            
            if res.status_code == 200:
                print(f"Status: SUCCESS ({duration} ms)")
                print(f"Completion:\n{res_data['completion']}\n")
                print(f"Ms/Token: {res_data['performance']['ms_per_token']} ms")
            else:
                print(f"Status: FAILED ({res.status_code}) -> {res.text}")
        except Exception as e:
            print(f"Request Error: {e}")
        print("-" * 50)

if __name__ == "__main__":
    run_benchmarks()