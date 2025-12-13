import os
import json
import httpx
import asyncio
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------
# SETUP
# ----------------------------------------------------------

mcp = FastMCP("127.0.0.1")

api_key = os.getenv("OPENROUTER_API_KEY")
url = "https://openrouter.ai/api/v1/chat/completions"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# Load embedding model once
embedder = SentenceTransformer("all-mpnet-base-v2")

# ----------------------------------------------------------
# LLM CALL WITH RETRY & BACKOFF
# ----------------------------------------------------------

@mcp.tool()
async def llm_call(prompt: str, retry=5):
    data = {
        "model": "openai/gpt-3.5-turbo",
        "messages": [{"role": "user", "content": prompt}]
    }

    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(retry):
            try:
                response = await client.post(url, headers=headers, json=data)

                # rate-limited
                if response.status_code == 429:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue

                return response

            except httpx.ReadTimeout:
                await asyncio.sleep(1.5 * (attempt + 1))

    return {"error": "LLM request failed after retries"}

# ----------------------------------------------------------
# SINGLE-CALL PARALLEL REASONING (Original)
# ----------------------------------------------------------

async def parallel_reasoning_single_call(prompt: str):
    """
    Single LLM call that generates 4 perspectives.
    More efficient but less independent reasoning.
    """
    multi_prompt = f"""
You are a parallel reasoning engine.

Generate 4 independent reasoning threads for the task below.
Each thread should think differently.
Return ONLY a JSON object with this structure:

{{
 "thread_step": "step-by-step reasoning",
 "thread_alt": "alternative reasoning path",
 "thread_flaw": "critical flaw analysis",
 "thread_creative": "creative or unexpected solution"
}}

Task: {prompt}
"""

    response = await llm_call(multi_prompt)

    # Case 1: retry failed → response is dict
    if isinstance(response, dict):
        return {
            "error": "LLM call failed after all retries",
            "raw": response
        }

    # Case 2: httpx response but content not parsed
    try:
        raw = response.json()
    except:
        return {
            "error": "Invalid JSON from LLM",
            "raw": str(response)
        }

    # Case 3: response exists but expected fields missing
    try:
        content = raw["choices"][0]["message"]["content"]
    except:
        return {
            "error": "Missing fields in LLM output",
            "raw": raw
        }

    # Case 4: JSON inside content fails to parse
    try:
        threads = json.loads(content)
    except:
        return {
            "error": "Failed to parse reasoning threads",
            "raw": content
        }

    return threads


# ----------------------------------------------------------
# TRUE PARALLEL MULTI-CALL REASONING (New)
# ----------------------------------------------------------

async def parallel_reasoning_multi_call(prompt: str):
    """
    4 independent LLM calls with different prompts.
    More expensive but truly parallel reasoning.
    """
    
    thread_prompts = {
        "thread_step": f"Provide step-by-step analysis:\n\n{prompt}",
        "thread_alt": f"Provide an alternative reasoning path:\n\n{prompt}",
        "thread_flaw": f"Analyze critical flaws or contradictions:\n\n{prompt}",
        "thread_creative": f"Provide a creative or unconventional solution:\n\n{prompt}"
    }
    
    # Run all 4 calls in parallel
    tasks = [
        llm_call(thread_prompts["thread_step"]),
        llm_call(thread_prompts["thread_alt"]),
        llm_call(thread_prompts["thread_flaw"]),
        llm_call(thread_prompts["thread_creative"])
    ]
    
    responses = await asyncio.gather(*tasks)
    
    threads = {}
    thread_names = ["thread_step", "thread_alt", "thread_flaw", "thread_creative"]
    
    for i, (name, response) in enumerate(zip(thread_names, responses)):
        if isinstance(response, dict) and "error" in response:
            threads[name] = "[ERROR in thread]"
            continue
        
        try:
            raw = response.json()
            content = raw["choices"][0]["message"]["content"]
            threads[name] = content
        except:
            threads[name] = "[ERROR parsing response]"
    
    return threads


# ----------------------------------------------------------
# MERGER FOR REASONING THREADS
# ----------------------------------------------------------

async def merge_threads(threads: dict):
    """Combine all reasoning threads into final answer."""
    merge_prompt = f"""
You are a reasoning synthesizer.

Merge the following reasoning threads into a single comprehensive explanation.
Identify agreements, contradictions, and insights from each perspective.

Threads:
{json.dumps(threads, indent=2)}

Return only the final synthesized explanation.
"""

    response = await llm_call(merge_prompt)

    if isinstance(response, dict):
        return "[ERROR: merge LLM failed]"

    try:
        raw = response.json()
        return raw["choices"][0]["message"]["content"]
    except:
        return "[ERROR: invalid merge response]"


# ----------------------------------------------------------
# UNIFIED REASONERS (CHOOSE ONE)
# ----------------------------------------------------------

async def unified_parallel_reasoner_single_best(prompt):
    threads = await parallel_reasoning_single_call(prompt)
    best = await select_most_confident_branch(threads, prompt)
    return best, 3  # +1 call for evaluation



async def unified_parallel_reasoner_multi_best(prompt):
    threads = await parallel_reasoning_multi_call(prompt)
    best = await select_most_confident_branch(threads, prompt)
    return best, 6  # +1 call for evaluation


# ----------------------------------------------------------
# CONFIDENCE EVALUATOR (NEW)
# ----------------------------------------------------------

async def evaluate_branch_confidence(branch_name:str, branch_text:str, task:str):
    """
    This function is used to evalute the epitimistic confidence of a reasoning branch.
    Returns a float score from 0 to 1.
    """
    eval_prompt = f"""
You are evaluating a reasoning attempt.

Task:
{task}
Reasoning branch ({branch_name}):{branch_text}

Score this branch on epistemic quality:
 internal consistency
 avoidance of hallucination
 acknowledgment of uncertainty where appropriate
 correctness relative to known theory

Return ONLY a JSON object:
{{ "confidence": <number between 0 and 1> }}
"""
    response = await llm_call(eval_prompt)

    if isinstance(response, dict):
        return 0.0
    
    try:
        raw = response.json()
        content = raw["choices"][0]["message"]["content"]
        score = json.loads(content)["confidence"]
        return float(score)
    except:
        return 0.0
    
# ----------------------------------------------------------
# Branch Selector
# ----------------------------------------------------------
async def select_most_confident_branch(threads:dict,task:str):
    """
    Evaluates all branches and gives the most confident one.
    """

    tasks = []
    for name, text in threads.items():
        tasks.append(evaluate_branch_confidence(name,text,task))
    scores = await asyncio.gather(*tasks)

    scored = list(zip(threads.keys(), threads.values(),scores))
    scored.sort(key = lambda x: x[2], reverse=True)

    best_name, best_text, best_score = scored[0]

    return {
        "branch": best_name,
        "confidence": best_score,
        "answer": best_text,
        "all_scores": {
            name: score for name, _, score in scored
        }
    }

async def evaluate_task_fulfillment(branch_text: str, task: str):
    """
    Scores whether the branch actually fulfills the task.
    """

    prompt = f"""
Evaluate whether the following response fulfills the task requirements.

Task:
{task}

Response:
{branch_text}

Does this response:
- directly address the task?
- attempt what is being asked?
- avoid sidestepping with summaries?

Return ONLY JSON:
{{ "fulfillment": <number between 0 and 1> }}
"""

    response = await llm_call(prompt)

    if isinstance(response, dict):
        return 0.0

    try:
        content = response.json()["choices"][0]["message"]["content"]
        return float(json.loads(content)["fulfillment"])
    except:
        return 0.0

async def select_best_branch_dual(threads: dict, task: str):
    results = []

    for name, text in threads.items():
        epistemic = await evaluate_branch_confidence(name, text, task)
        fulfillment = await evaluate_task_fulfillment(text, task)

        # weighted score
        final_score = 0.6 * epistemic + 0.4 * fulfillment

        results.append({
            "branch": name,
            "epistemic": epistemic,
            "fulfillment": fulfillment,
            "final": final_score,
            "answer": text
        })

    results.sort(key=lambda x: x["final"], reverse=True)
    return results[0], results

# ----------------------------------------------------------
# BENCHMARK RUNNER
# ----------------------------------------------------------

async def benchmark_both(prompt: str):
    """Run both approaches and compare."""
    
    print("\n" + "="*70)
    print("SINGLE-CALL APPROACH (1 prompt → 4 perspectives)")
    print("="*70)
    start = time.time()
    answer_single, calls_single = await unified_parallel_reasoner_single_best(prompt)
    time_single = time.time() - start
    print(f"Time: {time_single:.2f}s | API Calls: {calls_single}")
    print(f"Answer:\n{answer_single}\n")
    
    print("\n" + "="*70)
    print("MULTI-CALL APPROACH (4 independent prompts)")
    print("="*70)
    start = time.time()
    answer_multi, calls_multi = await unified_parallel_reasoner_multi_best(prompt)
    time_multi = time.time() - start
    print(f"Time: {time_multi:.2f}s | API Calls: {calls_multi}")
    print(f"Answer:\n{answer_multi}\n")
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"Single-call: {time_single:.2f}s ({calls_single} calls)")
    print(f"Multi-call:  {time_multi:.2f}s ({calls_multi} calls)")
    print(f"Speed difference: {time_multi/time_single:.2f}x slower (expected)")
    print(f"Cost difference: {calls_multi/calls_single:.2f}x more expensive")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

if __name__ == "__main__":
    # Test Case 1: Paradox (unsolvable)
    paradox_query = """A man says: "I am lying right now."
If his statement is true, it must be false.
If his statement is false, it must be true.

Explain what is happening and find the correct interpretation."""

    # Test Case 2: Logic Puzzle (has a definite solution)
    logic_puzzle_query = """
Prove or disprove P = NP
"""

    print("\n" + "#"*70)
    print("# TEST 1: PARADOX (Unsolvable - both should perform similarly)")
    print("#"*70)
    asyncio.run(benchmark_both(paradox_query))
    
    print("\n\n" + "#"*70)
    print("# TEST 2: LOGIC PUZZLE (Has definite solution - multi-call might excel)")
    print("#"*70)
    asyncio.run(benchmark_both(logic_puzzle_query))
    
    # Or run just one:
    # final, calls = asyncio.run(unified_parallel_reasoner_single(logic_puzzle_query))
    # final, calls = asyncio.run(unified_parallel_reasoner_multi(logic_puzzle_query))