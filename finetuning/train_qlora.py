"""QLoRA fine-tuning of Qwen2.5-Coder-3B-Instruct for Text-to-SQL on a Spider train subset.

Reads data/train_formatted.json (chat format: system / user[schema + question] / assistant[gold SQL]).
Settings that worked on a free Colab T4 (15 GB):
  - fp16 compute dtype for the 4-bit base; Trainer-level fp16/bf16 OFF (T4 has no fast bf16, and the
    GradScaler path crashed on mixed dtypes)
  - gradient checkpointing ON with use_reentrant=True (OFF -> CUDA OOM; non-reentrant -> CheckpointError with bnb 4-bit)
Usage: python train_qlora.py [--resume]
"""
import argparse, json, os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
OUT_DIR = os.path.join(ROOT, "qlora_checkpoints_v4")


def main(resume):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                             bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={"use_reentrant": True})
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    with open(os.path.join(ROOT, "data", "train_formatted.json")) as f:
        dataset = Dataset.from_list(json.load(f))

    args = SFTConfig(
        output_dir=OUT_DIR, per_device_train_batch_size=2, gradient_accumulation_steps=8,
        num_train_epochs=2, learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=10,
        logging_steps=10, save_steps=50, save_total_limit=3, max_length=1024,
        fp16=False, bf16=False, optim="paged_adamw_8bit",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": True},
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=dataset)
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(os.path.join(ROOT, "qlora_adapter_final"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume", action="store_true")
    main(p.parse_args().resume)
