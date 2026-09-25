# WRITEUP - RUN 2 ON SMALL THINKING DATASETS
2026-09-25
test case comparable with base instruct model.
BASE MODEL: `Qwen2.5-7B-Instruct`

## RESULTS

Trained on a filtered (thinking length below a certain amount of tokens, to reduce bloat), randomly selected batch of [1500 examples (FILE)](small-thoughts-1500.jsonl) from [SmallThoughts-10K](SmallThoughts-10K-raw.jsonl) with the system prompt ***included*** inside the training context.

Format quality improved drastically - the model reliably produces traces of `<|begin_of_thought|>` / `<|begin_of_solution|>` traces, gates thinking on the system prompt, and produces concise reasoning. However, code task performance ties with the base model on 8/10 tests.

Since both the base model and the fine-tune performed poorly, using a better fine tune of Qwen2.5 (e.g. Coder-Instruct) for coding and reasoning would be a better base. Another possible solution would be using more verified code examples in the dataset.

The system prompt was included in the training context (a mistake carried over from the previous pipeline, not intended). This caused the model to gate thinking on the system prompt - a behavior we will remove in Run 3.

Diagnosis: the training data is predominantly math, and its verification pattern is symbolic (algebraic substitution), not execution-based (tracing code on concrete inputs). The model learned the shape of thinking but not code verification. Additionally, including the system prompt during training caused the model to gate thinking on it, which was not the intended behavior and it affects responses in a bad way. The next experiment will revert `format_row` to user-only and mix in code verification traces.

## DATA

DATA TRAINED ON: [small-thoughts-1500.jsonl](small-thoughts-1500.jsonl)  
1500 examples after choosing.

SOURCE DATA:  
[small-thoughts-10K-raw.jsonl](small-thoughts-10K-raw.jsonl)

Filtered by character length inside the <|begin_of_thought|> block (threshold: 800 characters); examples exceeding it were discarded. The 1500 batch was chosen *randomly*, which could've caused issues. The approach of the next experiment will take that into account and filter them by example type and quality.

## TESTS
Check the questions and responses [here (comparison_results.txt)](comparison_results.txt).
Due to the necessary prompt, the bad results are visible - the answer to "What is the capital of France" is way longer than it needs to be for both models.
