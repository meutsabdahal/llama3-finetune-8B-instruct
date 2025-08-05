# Fine-Tuning LLaMA 3 8B Instruct with QLoRA on HumanEval

This repository contains code and configuration for fine-tuning the `meta-llama/Meta-Llama-3-8B-Instruct` model using QLoRA on the [HumanEval dataset](https://huggingface.co/datasets/openai_humaneval).

## Objective

Fine-tune LLaMA 3 8B for code generation using OpenAI's HumanEval benchmark — a dataset of prompts and canonical Python solutions used for evaluating AI-generated code.

## Dataset

* **Name:** `openai_humaneval`
* **Source:** Hugging Face Datasets Hub
* **Split Used:** `test`
* **Format:** Each example consists of a code `prompt` and its `canonical_solution`. We concatenate both into a single `text` field for supervised fine-tuning.

## Model

* **Base Model:** `meta-llama/Meta-Llama-3-8B-Instruct`
* **Tokenizer:** Loaded via `AutoTokenizer`
* **Quantization:** 4-bit using `bitsandbytes` (nf4 quantization, double quant enabled)
* **LoRA Configuration:**

  * Rank: 8
  * Alpha: 32
  * Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`
  * Dropout: 0.05

## Training

* **Trainer:** `SFTTrainer` from `trl`
* **Batch Size:** 1 (with gradient accumulation)
* **Epochs:** 2
* **Learning Rate:** 2e-4
* **Precision:** bfloat16
* **Saving:** Checkpoints saved at each epoch

## Output

Adapter model and tokenizer are saved to:

```
/content/drive/MyDrive/llama3-finetune-8B-instruct/outputs/final_adapter/
```

You can load them using `peft` for inference.

## Requirements

```bash
pip install transformers datasets accelerate bitsandbytes peft trl
```

## Run Training (Colab Compatible)

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    args=sft_args,
    tokenizer=tokenizer
)

trainer.train()
trainer.model.save_pretrained("path/to/save")
tokenizer.save_pretrained("path/to/save")
```

---

## Notes

* Training was done using Google Colab with a T4 GPU (14.74 GB VRAM).
* To avoid memory issues, model is quantized to 4-bit with QLoRA.
* The project does **not** save the full model, only LoRA adapters.
