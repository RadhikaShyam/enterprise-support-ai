import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from peft import LoraConfig
from trl import SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TRAIN_FILE = (
    "data/fine_tuning/train/train.jsonl"
)

VALIDATION_FILE = (
    "data/fine_tuning/validation/validation.jsonl"
)

OUTPUT_DIR = "outputs/qlora/support-model"


def main():

    print("=" * 60)
    print("QLoRA TRAINING")
    print("=" * 60)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )

    # --------------------------------------------------
    # 1. Load tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # 2. 4-bit quantization
    # --------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------
    # 3. Load model
    # --------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # --------------------------------------------------
    # 4. Load datasets
    # --------------------------------------------------

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALIDATION_FILE,
        },
    )

    print(
        "Training examples:",
        len(dataset["train"]),
    )

    print(
        "Validation examples:",
        len(dataset["validation"]),
    )

    # --------------------------------------------------
    # 5. Convert our instruction format into text
    # --------------------------------------------------

    def format_example(example):

        return (
            "<|system|>\n"
            "You are an enterprise IT support assistant. "
            "Answer accurately using the available support "
            "knowledge.\n"
            "<|user|>\n"
            f"{example['instruction']}\n"
            "<|assistant|>\n"
            f"{example['output']}"
        )

    # --------------------------------------------------
    # 6. LoRA configuration
    # --------------------------------------------------

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    # --------------------------------------------------
    # 7. Training configuration
    # --------------------------------------------------

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=1,

        per_device_eval_batch_size=1,

        gradient_accumulation_steps=8,

        gradient_checkpointing=True,

        learning_rate=2e-4,

        fp16=False,
        
        bf16=True,

        logging_steps=1,

        eval_strategy="epoch",

        save_strategy="epoch",

        save_total_limit=2,

        report_to="none",

        optim="paged_adamw_8bit",

        weight_decay=0.01,
    )

    # --------------------------------------------------
    # 8. Trainer
    # --------------------------------------------------

    trainer = SFTTrainer(
        model=model,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        processing_class=tokenizer,

        peft_config=peft_config,

        formatting_func=format_example,

    )

    # --------------------------------------------------
    # 9. Train
    # --------------------------------------------------

    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    trainer.train()

    # --------------------------------------------------
    # 10. Save adapter
    # --------------------------------------------------

    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        "Adapter saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()