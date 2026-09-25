# WRITEUP
2026-09-18
TEST CASE FAILURE - DOES NOT PERFORM AS GOOD AS BASE

### RESULTS

After experimenting, the model seemingly performed better at thinking than its base, but it has major drawbacks. Noted results;

#### DRAWBACKS & NEGATIVES

- Writes full code 1 or more times inside its `<think>` brackets, then writes it again in the answer. - **POSSIBLE CAUSE:** _training data has examples where that happens, it learnt from there_
    
- Sidetracks from the actual prompt -- ignores the main question and starts over-explaining unprompted topics. Gets distracted and over-explains.
    
- Bugs -- writes down wrong code, with the wrong fixes. Obvious mistakes get written as an answer.  
    EXAMPLE: `amount > self.balance` is correct when withdrawing money. Fine-tune spits out: `amount >= self.balance` - so if you attempt to withdraw your exact sum, it will throw an exception. _CRITICAL MISTAKE - the reason why this experiment's status went to a test case failure._
    

#### POSITIVES

- Thinking is properly done & short. Does not bloat unnecessarily _too much._  
    **NOTE:** _the model still overdoes thinking, but this alone would not make the experiment a fail._
    
- When correct, the fine-tune's code often includes better edge case handling and helper methods than the base. However, this is inconsistent and often buried under duplication.
    

### HYPOTHESIS

Data quality is the main culprit of the intense drawbacks. The mini-reasoning could be faulty, the dataset needs more filtering and more data.

**Suggested fixes:** research more dataset types, mix them with better ratios. `qlora_1500` examples is faulty, because it caused bloat on its on in previous experiments. Either heavily filter it next time or completely discard it and replace with a better thinking set.

**Next steps:**  
Improve on data only, keep using the same training algorithm. 20-30% performance increase is expected after all of the examples are good. The reason this failed is because the dataset is unsanitized. Training algorithm, setup and parameters is presumed to be good for the time being.

The goals for the next test case are narrow down possible mistakes in the data and other factors that ruin model performance.

A test case will be determined a success and publish-worthy for the paper once a fine-tuned model can confidently outperform its derivative model.

> **ADDITIONAL INFO**
> Model weights are not published. This run was a negative result and the weights are not useful for downstream use. A future run will release weights if it outperforms the base.

### DATA

DATA TRAINED ON: [mixed_reasoning_v1.jsonl](mixed_reasoning_v1.jsonl)  
521 examples after filtering.

SOURCE DATA:  
[mini_reasoning_clean_v2.jsonl](mini_reasoning_clean_v2.jsonl) + [qlora_reasoning_1500.jsonl](qlora_reasoning_1500.jsonl)

HOW ITS FILTERED:

MINI REASONING -> stripped of all answers that do not have an answer after `<think>` tags. Reasoning is short and proper, as intended.

QLORA 1500 -> randomly picked about 200 examples with longer thinking, since the questions there were more complex.

Combined the mini reasoning + the 200 examples into `mixed_reasoning_v1`.

### FINE-TUNING

Fine tuning took about ~1 hour on Colab.

**CODE USED FOR FINE-TUNING:** [qlora_tuning_2.ipynb](qlora_tuning_2.ipynb)

#### PARAMS

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

*dataset loading*
```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="mixed_reasoning_v1.jsonl", split="train")

print(dataset)

# Apply Qwen chat template
def format_row(row):
    return {"text": tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )}
    
dataset = dataset.map(format_row)
print(dataset[0]["text"][:500])
```

*training params*
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        max_seq_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=1e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
    ),
)
```
