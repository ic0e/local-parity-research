import json
import os
import time
from typing import List
from groq import Groq, RateLimitError, APIError
from pydantic import BaseModel, Field

API_KEY = os.environ.get("GROQ_API_KEY", "your-groq-api-key-here")
OUTPUT_FILE = "dataset_examples.json"
MODEL = "openai/gpt-oss-120b"
TARGET_COUNT = 500

client = Groq(api_key=API_KEY)

class ExampleSchema(BaseModel):
    instruction: str
    response: str
    category: str = Field(default="Reasoning")

def load_existing_data(filepath: str) -> list:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
    return []

def save_dataset(filepath: str, data: list) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_data():
    dataset = load_existing_data(OUTPUT_FILE)
    print(f"Starting run. Existing examples: {len(dataset)}")

    system_prompt = (
        "You are an expert AI dataset curator specializing in synthetic Chain-of-Thought (CoT) and reasoning datasets. "
        "Your task is to generate high-quality dataset entries designed to train open-weights models to think, analyze, and reason step-by-step in JSON format."
    )

    user_prompt = (
        "You are a high-quality training data generator for fine-tuning a reasoning model on PRACTICAL CODING PROBLEMS.\n\n"
        
        "CORE MISSION:\n"
        "Generate 1 diverse, real-world coding problem that reflects actual development work.\n"
        "The problem should teach a junior developer something they'd encounter in production.\n"
        "Focus on: implementation, debugging, optimization, and architectural reasoning.\n\n"
        
        "DIFFICULTY:\n"
        "Randomly select ONE difficulty (EASY, MEDIUM, or HARD) and generate a problem fitting that level:\n"
        "- EASY → Fundamental concept with practical twist (data structure, basic algorithm, common pattern)\n"
        "- MEDIUM → Real bug fix or optimization challenge\n"
        "- HARD → Complex reasoning (algorithm design, system trade-offs, performance analysis)\n\n"
        
        "PROBLEM REQUIREMENTS:\n"
        "Every problem MUST include:\n"
        "1. REAL-WORLD CONTEXT: What's the actual use case? Scale of data? Performance needs?\n"
        "   Examples:\n"
        "   - 'Your API serves 10K requests/sec and needs sub-100ms pagination'\n"
        "   - 'Your cache is consuming 80% of memory; optimize it for 10M keys'\n"
        "   - 'Production bug: sorting fails silently on duplicate IDs in 50M record dataset'\n"
        "2. SPECIFIC CONSTRAINTS: Don't say \"write efficient code\"; say \"O(n) time, O(1) space\"\n"
        "3. EDGE CASES: Mention what happens at scale, with bad input, with empty data\n"
        "4. WHY IT MATTERS: Connect to real problems (latency, memory, correctness, maintainability)\n\n"
        
        "RESPONSE FORMAT (STRICT):\n"
        "Every response must follow this exact structure:\n\n"
        "<think>\n"
        "[Meticulous step-by-step reasoning:\n"
        "1. Understand the problem and constraints clearly\n"
        "2. Identify edge cases and failure modes\n"
        "3. Analyze alternatives (why pick this approach over others?)\n"
        "4. State time and space complexity\n"
        "5. Verify correctness with examples\n"
        "Keep this concise: 200-300 tokens, not verbose]\n"
        "</think>\n\n"
        "[Clear, production-ready final response:\n"
        "- Working code with inline comments for non-obvious logic\n"
        "- Input validation and error handling\n"
        "- Trade-off explanation (what you're optimizing for and what you're sacrificing)\n"
        "- When/why this approach breaks down at different scales]\n\n"
        
        "CATEGORY OPTIONS:\n"
        "Use one of these; they reflect real development work:\n"
        "- 'Code Implementation'\n"
        "- 'Code Debugging'\n"
        "- 'Code Optimization'\n"
        "- 'Algorithm Design'\n"
        "- 'System Architecture'\n"
        "- 'Performance Analysis'\n\n"
        
        "QUALITY VERIFICATION CHECKLIST:\n"
        "Before finalizing each problem, ask:\n"
        "□ Can a junior dev understand this and learn something useful?\n"
        "□ Does the solution actually work (would it pass code review)?\n"
        "□ Is the reasoning concise but complete (200-300 tokens for <think>)?\n"
        "□ Is this something you'd encounter in real production code?\n"
        "□ Are edge cases handled and explained?\n"
        "□ Are trade-offs explicitly stated?\n"
        "□ Would this code actually run without errors?\n"
        "□ Is input validation defensive?\n\n"
        
        "OUTPUT FORMAT (STRICT JSON):\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"instruction\": \"Detailed problem statement with real-world context and constraints\",\n"
        "      \"response\": \"<think>\\n[reasoning 200-300 tokens]\\n</think>\\n\\n[final response with code]\",\n"
        "      \"category\": \"Code Implementation\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        
        "REASONING TEMPLATE (for <think> block):\n"
        "1. [Clarify inputs and outputs: what exactly are we solving?]\n"
        "2. [List edge cases: empty input, boundary values, scale limits, invalid inputs]\n"
        "3. [Consider alternatives: why this approach over others?]\n"
        "4. [Complexity analysis: state time and space clearly]\n"
        "5. [Verify with examples: does this work on edge cases?]\n"
        "Keep each point to 1-2 sentences. Total: 200-300 tokens.\n\n"
        
        "RESPONSE TEMPLATE (for final answer):\n"
        "[Brief intro: what problem we're solving and why]\n\n"
        "[Code block with comments]\n\n"
        "[Complexity analysis: O(?) time, O(?) space]\n\n"
        "[Trade-offs: what are we optimizing for? What's the cost?]\n\n"
        "[When it breaks: at what scale/conditions does this fail?]\n\n"
        "[Why this approach: why is this better than alternatives?]\n\n"
        
        "GENERATE NOW:\n"
        "Create 1 problem following the above guidelines.\n"
        "Output valid JSON. The problem must be production-grade quality."
    )

    while len(dataset) < TARGET_COUNT:
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
                max_tokens=4096,
            )

            raw_json = completion.choices[0].message.content
            parsed_json = json.loads(raw_json)

            raw_items = parsed_json if isinstance(parsed_json, list) else parsed_json.get("items", [])
            
            valid_items = []
            for item in raw_items:
                validated = ExampleSchema.model_validate(item).model_dump()
                if "<think>" in validated["response"] and "</think>" in validated["response"]:
                    valid_items.append(validated)

            dataset.extend(valid_items)
            save_dataset(OUTPUT_FILE, dataset)
            
            print(f"Total collected: {len(dataset)} / {TARGET_COUNT} (+{len(valid_items)} added)")
            time.sleep(1.5)

        except RateLimitError as e:
            retry_after = getattr(e, "response", {}).headers.get("retry-after") if hasattr(e, "response") else None
            wait_time = float(retry_after) + 1.0 if retry_after else 10.0
            print(f"Rate limit hit. Sleeping {wait_time:.1f}s...")
            time.sleep(wait_time)

        except (APIError, Exception) as e:
            print(f"Error: {e}")
            save_dataset(OUTPUT_FILE, dataset)
            time.sleep(2.0)
            continue

    print(f"\nFinished! Saved {len(dataset)} examples to '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    generate_data()
