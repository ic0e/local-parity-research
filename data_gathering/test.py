import json
import os
import pathlib
from groq import Groq
from pydantic import BaseModel, Field

class SingleTestEntry(BaseModel):
    instruction: str = Field(description="The problem statement")
    reasoning_trace: str = Field(description="Step-by-step thinking trace")
    final_output: str = Field(description="The verified code or answer")

def run_groq_test():
    script_dir = pathlib.Path(__file__).parent.resolve()
    output_path = script_dir / "test_groq_output.json"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        print('In PowerShell, set it using: $env:GROQ_API_KEY="gsk_your_key"')
        return

    client = Groq(api_key=api_key)

    # Active Groq models (2026)
    active_models = [
        "openai/gpt-oss-120b",
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile"
    ]

    # Explicitly demand exact schema keys in the prompt
    system_prompt = """
    You are an expert dataset generator. You MUST respond ONLY with a raw JSON object containing EXACTLY these 3 keys:
    1. "instruction": A clear, complex prompt or task description.
    2. "reasoning_trace": Step-by-step logical breakdown, edge-case analysis, or pseudocode execution.
    3. "final_output": The exact code snippet, answer, or final result.

    Do not use markdown blocks like ```json or add extra keys like 'status' or 'topic'.
    """

    user_prompt = "Generate a synthetic training entry on: 'Python List Comprehension Edge Cases'."

    for model_name in active_models:
        print(f"Connecting to Groq with model: {model_name}...")
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )

            raw_json = response.choices[0].message.content.strip()
            parsed_data = json.loads(raw_json)

            # Defensive key mapping fallback in case model aliases keys
            instruction = parsed_data.get("instruction") or parsed_data.get("prompt") or user_prompt
            reasoning = parsed_data.get("reasoning_trace") or parsed_data.get("reasoning") or parsed_data.get("thinking") or "No trace provided"
            final_out = parsed_data.get("final_output") or parsed_data.get("output") or parsed_data.get("answer") or "No output provided"

            validated_entry = SingleTestEntry(
                instruction=instruction,
                reasoning_trace=reasoning,
                final_output=final_out
            )

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(validated_entry.model_dump(), f, indent=2, ensure_ascii=False)

            print(f"\n[SUCCESS] Generated dataset entry using '{model_name}'!")
            print(f"Saved to: {output_path}\n")
            return

        except Exception as e:
            print(f"Model '{model_name}' failed: {e}\n")

    print("All models failed. Check your API key or network connection.")

if __name__ == "__main__":
    run_groq_test()
