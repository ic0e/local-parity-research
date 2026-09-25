# WRITEUP - RUN 2 ON SMALL THINKING DATASETS
2026-09-25
test case comparable with base instruct model.
BASE MODEL: `Qwen2.5-7B-Instruct`

## RESULTS

Trained on a filtered (thinking length below a certain amount of tokens, to reduce bloat), randomly selected batch of [1500 examples (FILE)](small-thoughts-1500.jsonl) from [SmallThoughts-10K](SmallThoughts-10K-raw.jsonl) with the system prompt ***included*** inside the training context.

Format quality improved drastically - the model reliably produces traces of `<|begin_of_thought|>` / `<|begin_of_solution|>` traces, gates thinking on the system prompt, and produces concise reasoning. However, code task performance ties with the base model on 8/10 tests.

Since both the base model and the fine-tune performed poorly, using a better fine tune of Qwen2.5 (e.g. Coder-Instruct) for coding and reasoning would be a better base. Another possible solution would be using more verified code examples in the dataset.

Diagnosis: the training data is predominantly math, and its verification pattern is symbolic (algebraic substitution), not execution-based (tracing code on concrete inputs). The model learned the shape of thinking but not code verification. Additionally, including the system prompt during training caused the model to gate thinking on it, which was not the intended behavior. The next experiment will revert `format_row` to user-only and mix in code verification traces.

## DATA

DATA TRAINED ON: [small-thoughts-1500.jsonl](small-thoughts-1500.jsonl)  
1500 examples after choosing.

SOURCE DATA:  
[small-thoughts-10K-raw.jsonl](small-thoughts-10K-raw.jsonl)

Filtered by token length inside thinking brackets, examples that are too long were trimmed.
The 1500 batch was chosen *randomly*, which could've caused issues. The approach of the next experiment will take that into account and filter them by example type and quality.
