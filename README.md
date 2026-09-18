# QLoRA Reasoning Experiments - Local Parity Research

This is a repository where I document my experiments with fine tuning small models on reasoning data. The **current** goal is to see if curated short CoT data can improve a base model's performance on tasks like math, logic, and code generation.

Each experiment has its own folder with the dataset used, the training config, and a writeup of the results. Writeups include the hypothesis, the failure modes, and more information. (lab notebook style) Documented so the results are reproducible.

## Experiments
- [001 - Qwen7B-QLoRA](experiments/001-Qwen7B-QLoRA) [**Model Fail HIGH SEVERITY**] - The first experiment, writeup is still WIP due to it not being a priority. Data examples: ~1500.
- [002 - qlora_mixed_reasoning_v1](experiments/qlora_mixed_reasoning_v1/) [**Model Fail MEDIUM SEVERITY**] - Base model wins on code generation. Fine tune introduces duplication and reasoning bugs. A drastic improvement from 001, with less data examples (~500).
