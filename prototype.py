import os
import json
import httpx
import asyncio
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------

@dataclass
class Config:
    """Centralized configuration."""
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model: str = "openai/gpt-3.5-turbo"
    url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 30
    max_retries: int = 5
    backoff_base: float = 1.5
    
    # Scoring weights
    epistemic_weight: float = 0.6
    fulfillment_weight: float = 0.4

config = Config()

# ----------------------------------------------------------
# SETUP
# ----------------------------------------------------------

mcp = FastMCP("127.0.0.1")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {config.api_key}"
}

# Load embedding model (for potential future semantic comparison)
embedder = SentenceTransformer("all-mpnet-base-v2")

# ----------------------------------------------------------
# ENHANCED LLM CALL WITH RETRY & BACKOFF
# ----------------------------------------------------------

@mcp.tool()
async def llm_call(
    prompt: str, 
    temperature: float = 0.7,
    max_tokens: int = 1000,
    retry: int = None
) -> Dict:
    """
    Robust LLM API call with exponential backoff.
    
    Returns:
        Dict with 'content' on success or 'error' on failure
    """
    retry = retry or config.max_retries
    
    data = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=config.timeout) as client:
        for attempt in range(retry):
            try:
                response = await client.post(config.url, headers=headers, json=data)
                
                # Handle rate limiting
                if response.status_code == 429:
                    wait_time = config.backoff_base ** (attempt + 1)
                    print(f"⏳ Rate limited, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                
                # Parse successful response
                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    return {"content": content, "status": "success"}
                
                # Other errors
                return {"error": f"HTTP {response.status_code}", "status": "failed"}

            except httpx.ReadTimeout:
                wait_time = config.backoff_base ** (attempt + 1)
                print(f"⏳ Timeout, retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                return {"error": str(e), "status": "failed"}

    return {"error": "Max retries exceeded", "status": "failed"}


# ----------------------------------------------------------
# SINGLE-CALL PARALLEL REASONING
# ----------------------------------------------------------

async def parallel_reasoning_single_call(prompt: str) -> Dict:
    """
    Single LLM call generating 4 perspectives internally.
    Efficient but less independent.
    """
    multi_prompt = f"""You are a parallel reasoning engine.

Generate 4 independent reasoning threads for the task below.
Each thread must use different logic, assumptions, or approaches.

Return ONLY a valid JSON object (no markdown, no extra text):

{{
  "thread_step": "step-by-step analytical reasoning",
  "thread_alt": "alternative reasoning path with different assumptions",
  "thread_flaw": "critical flaw analysis and adversarial critique",
  "thread_creative": "creative or unconventional solution"
}}

Task: {prompt}"""

    response = await llm_call(multi_prompt, temperature=0.8)

    # Handle API failures
    if response.get("status") == "failed":
        return {
            "error": "LLM call failed",
            "details": response.get("error"),
            "threads": {
                "thread_step": "[ERROR]",
                "thread_alt": "[ERROR]",
                "thread_flaw": "[ERROR]",
                "thread_creative": "[ERROR]"
            }
        }

    # Parse JSON response
    try:
        content = response["content"].strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        threads = json.loads(content)
        
        # Validate expected keys
        required = ["thread_step", "thread_alt", "thread_flaw", "thread_creative"]
        if not all(k in threads for k in required):
            raise ValueError("Missing required thread keys")
        
        return threads

    except (json.JSONDecodeError, ValueError) as e:
        return {
            "error": "Failed to parse reasoning threads",
            "details": str(e),
            "raw_content": response.get("content", ""),
            "threads": {
                "thread_step": response.get("content", "[PARSE ERROR]"),
                "thread_alt": "[PARSE ERROR]",
                "thread_flaw": "[PARSE ERROR]",
                "thread_creative": "[PARSE ERROR]"
            }
        }


# ----------------------------------------------------------
# TRUE PARALLEL MULTI-CALL REASONING
# ----------------------------------------------------------

async def parallel_reasoning_multi_call(prompt: str) -> Dict:
    """
    4 independent LLM calls with different prompts.
    More expensive but truly parallel reasoning.
    """
    
    thread_prompts = {
        "thread_step": f"""Provide detailed step-by-step analytical reasoning.
Break down the problem logically and systematically.

Task: {prompt}""",
        
        "thread_alt": f"""Provide an alternative reasoning path.
Use different assumptions or a completely different approach.

Task: {prompt}""",
        
        "thread_flaw": f"""Analyze critical flaws, edge cases, and contradictions.
Be adversarial—what could go wrong with typical solutions?

Task: {prompt}""",
        
        "thread_creative": f"""Provide a creative, unconventional, or lateral solution.
Think outside standard approaches.

Task: {prompt}"""
    }
    
    # Launch all 4 calls in parallel
    tasks = [
        llm_call(thread_prompts["thread_step"], temperature=0.5),
        llm_call(thread_prompts["thread_alt"], temperature=0.8),
        llm_call(thread_prompts["thread_flaw"], temperature=0.7),
        llm_call(thread_prompts["thread_creative"], temperature=0.9)
    ]
    
    responses = await asyncio.gather(*tasks)
    
    # Collect results
    threads = {}
    thread_names = ["thread_step", "thread_alt", "thread_flaw", "thread_creative"]
    
    for name, response in zip(thread_names, responses):
        if response.get("status") == "failed":
            threads[name] = f"[ERROR: {response.get('error', 'unknown')}]"
        else:
            threads[name] = response.get("content", "[EMPTY RESPONSE]")
    
    return threads


# ----------------------------------------------------------
# EPISTEMIC CONFIDENCE EVALUATOR
# ----------------------------------------------------------

async def evaluate_branch_confidence(
    branch_name: str, 
    branch_text: str, 
    task: str
) -> float:
    """
    Scores epistemic quality of a reasoning branch.
    
    Returns:
        Float ∈ [0, 1] representing confidence
    """
    eval_prompt = f"""You are an epistemic evaluator for reasoning quality.

**Task**:
{task}

**Reasoning branch ({branch_name})**:
{branch_text}

Evaluate this reasoning on:
1. Internal consistency (no contradictions)
2. Avoidance of hallucination or unfounded claims
3. Appropriate acknowledgment of uncertainty
4. Correctness relative to established knowledge

Return ONLY a JSON object (no markdown):
{{ "confidence": <float between 0.0 and 1.0>, "reasoning": "<brief justification>" }}"""

    response = await llm_call(eval_prompt, temperature=0.3)

    if response.get("status") == "failed":
        print(f"⚠️  Confidence evaluation failed for {branch_name}")
        return 0.0
    
    try:
        content = response["content"].strip()
        
        # Clean markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        confidence = float(result["confidence"])
        
        # Clamp to valid range
        return max(0.0, min(1.0, confidence))
    
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"⚠️  Failed to parse confidence for {branch_name}: {e}")
        return 0.0


# ----------------------------------------------------------
# TASK FULFILLMENT EVALUATOR
# ----------------------------------------------------------

async def evaluate_task_fulfillment(branch_text: str, task: str) -> float:
    """
    Scores whether the branch actually addresses the task.
    
    Returns:
        Float ∈ [0, 1] representing fulfillment
    """
    prompt = f"""Evaluate whether this response fulfills the task requirements.

**Task**:
{task}

**Response**:
{branch_text}

Does this response:
- Directly address the task?
- Attempt what is being asked?
- Avoid deflection or summary without substance?

Return ONLY JSON (no markdown):
{{ "fulfillment": <float between 0.0 and 1.0>, "reasoning": "<brief justification>" }}"""

    response = await llm_call(prompt, temperature=0.3)

    if response.get("status") == "failed":
        return 0.0

    try:
        content = response["content"].strip()
        
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        result = json.loads(content)
        return max(0.0, min(1.0, float(result["fulfillment"])))
    
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0.0


# ----------------------------------------------------------
# BRANCH SELECTORS
# ----------------------------------------------------------

async def select_most_confident_branch(threads: Dict, task: str) -> Dict:
    """
    Select branch with highest epistemic confidence.
    """
    # Evaluate all branches in parallel
    tasks = [
        evaluate_branch_confidence(name, text, task)
        for name, text in threads.items()
        if not text.startswith("[ERROR")
    ]
    
    valid_threads = [
        (name, text) for name, text in threads.items()
        if not text.startswith("[ERROR")
    ]
    
    if not valid_threads:
        return {
            "error": "All threads failed",
            "branch": "none",
            "confidence": 0.0,
            "answer": "[ALL THREADS FAILED]"
        }
    
    scores = await asyncio.gather(*tasks)
    
    # Pair threads with scores
    scored = list(zip(
        [name for name, _ in valid_threads],
        [text for _, text in valid_threads],
        scores
    ))
    scored.sort(key=lambda x: x[2], reverse=True)
    
    best_name, best_text, best_score = scored[0]
    
    return {
        "branch": best_name,
        "confidence": best_score,
        "answer": best_text,
        "all_scores": {name: score for name, _, score in scored}
    }


async def select_best_branch_dual(threads: Dict, task: str) -> Tuple[Dict, List[Dict]]:
    """
    Select branch using dual scoring: epistemic + task fulfillment.
    
    Returns:
        (best_branch, all_results_sorted)
    """
    valid_threads = [
        (name, text) for name, text in threads.items()
        if not text.startswith("[ERROR")
    ]
    
    if not valid_threads:
        return {
            "error": "All threads failed",
            "branch": "none",
            "final": 0.0
        }, []
    
    results = []
    
    for name, text in valid_threads:
        epistemic_task = evaluate_branch_confidence(name, text, task)
        fulfillment_task = evaluate_task_fulfillment(text, task)
        
        epistemic, fulfillment = await asyncio.gather(epistemic_task, fulfillment_task)
        
        # Weighted combination
        final_score = (
            config.epistemic_weight * epistemic + 
            config.fulfillment_weight * fulfillment
        )
        
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
# UNIFIED REASONERS
# ----------------------------------------------------------

async def unified_parallel_reasoner_single_best(prompt: str) -> Tuple[Dict, int]:
    """Single-call with epistemic selection."""
    threads = await parallel_reasoning_single_call(prompt)
    
    if "error" in threads:
        return threads, 1
    
    best = await select_most_confident_branch(threads, prompt)
    return best, 5  # 1 generation + 4 evaluations


async def unified_parallel_reasoner_multi_best(prompt: str) -> Tuple[Dict, int]:
    """Multi-call with epistemic selection."""
    threads = await parallel_reasoning_multi_call(prompt)
    best = await select_most_confident_branch(threads, prompt)
    return best, 8  # 4 generation + 4 evaluations


async def unified_parallel_reasoner_dual(prompt: str, multi_call: bool = False) -> Tuple[Dict, List[Dict], int]:
    """Dual-score selection (epistemic + fulfillment)."""
    if multi_call:
        threads = await parallel_reasoning_multi_call(prompt)
        calls = 12  # 4 gen + 4 epistemic + 4 fulfillment
    else:
        threads = await parallel_reasoning_single_call(prompt)
        if "error" in threads:
            return threads, [], 1
        calls = 9  # 1 gen + 4 epistemic + 4 fulfillment
    
    best, all_results = await select_best_branch_dual(threads, prompt)
    return best, all_results, calls


# ----------------------------------------------------------
# BENCHMARK RUNNER
# ----------------------------------------------------------

async def benchmark_both(prompt: str, use_dual_scoring: bool = False):
    """Compare single-call vs multi-call approaches."""
    
    print("\n" + "="*70)
    print("🔹 SINGLE-CALL APPROACH (1 prompt → 4 perspectives)")
    print("="*70)
    start = time.time()
    
    if use_dual_scoring:
        answer_single, all_single, calls_single = await unified_parallel_reasoner_dual(prompt, multi_call=False)
    else:
        answer_single, calls_single = await unified_parallel_reasoner_single_best(prompt)
    
    time_single = time.time() - start
    
    print(f"⏱️  Time: {time_single:.2f}s | 📞 API Calls: {calls_single}")
    print(f"\n🏆 Best Branch: {answer_single.get('branch', 'N/A')}")
    if 'confidence' in answer_single:
        print(f"🎯 Confidence: {answer_single['confidence']:.3f}")
    if 'final' in answer_single:
        print(f"📊 Final Score: {answer_single['final']:.3f}")
    print(f"\n💬 Answer:\n{answer_single.get('answer', 'ERROR')[:500]}...\n")
    
    print("\n" + "="*70)
    print("🔹 MULTI-CALL APPROACH (4 independent prompts)")
    print("="*70)
    start = time.time()
    
    if use_dual_scoring:
        answer_multi, all_multi, calls_multi = await unified_parallel_reasoner_dual(prompt, multi_call=True)
    else:
        answer_multi, calls_multi = await unified_parallel_reasoner_multi_best(prompt)
    
    time_multi = time.time() - start
    
    print(f"⏱️  Time: {time_multi:.2f}s | 📞 API Calls: {calls_multi}")
    print(f"\n🏆 Best Branch: {answer_multi.get('branch', 'N/A')}")
    if 'confidence' in answer_multi:
        print(f"🎯 Confidence: {answer_multi['confidence']:.3f}")
    if 'final' in answer_multi:
        print(f"📊 Final Score: {answer_multi['final']:.3f}")
    print(f"\n💬 Answer:\n{answer_multi.get('answer', 'ERROR')[:500]}...\n")
    
    print("\n" + "="*70)
    print("📊 COMPARISON SUMMARY")
    print("="*70)
    print(f"Single-call: {time_single:.2f}s ({calls_single} calls)")
    print(f"Multi-call:  {time_multi:.2f}s ({calls_multi} calls)")
    print(f"Speed ratio: {time_multi/time_single:.2f}× slower")
    print(f"Cost ratio:  {calls_multi/calls_single:.2f}× more expensive")
    print("="*70)


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    """Run benchmark experiments."""
    
    # Test Case 1: Paradox (no definite solution)
    paradox_query = """A man says: "I am lying right now."

If his statement is true, then he is lying, which makes it false.
If his statement is false, then he is not lying, which makes it true.

Explain what is happening and resolve this paradox."""

    # Test Case 2: Hard problem (definite but complex)
    hard_problem = """Prove or disprove: P = NP

Provide reasoning for your position."""

    # Test Case 3: Practical reasoning
    practical_query = """You have 3 boxes. Box A contains 2 red balls, Box B contains 1 red and 1 blue ball, 
Box C contains 2 blue balls. You randomly pick a box and draw a ball—it's red. 

What's the probability the other ball in that box is also red?"""

    print("\n" + "#"*70)
    print("# PARALLEL REASONING BENCHMARK - Enhanced Version")
    print("#"*70)
    
    print("\n" + "#"*70)
    print("# TEST 1: PARADOX (No objective solution)")
    print("#"*70)
    asyncio.run(benchmark_both(paradox_query, use_dual_scoring=False))
    
    print("\n\n" + "#"*70)
    print("# TEST 2: HARD PROBLEM (Requires careful reasoning)")
    print("#"*70)
    asyncio.run(benchmark_both(hard_problem, use_dual_scoring=True))
    
    print("\n\n" + "#"*70)
    print("# TEST 3: PROBABILITY PUZZLE (Definite answer)")
    print("#"*70)
    asyncio.run(benchmark_both(practical_query, use_dual_scoring=True))


if __name__ == "__main__":
    main()
