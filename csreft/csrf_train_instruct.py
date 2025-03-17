import os
import sys
import json
import logging
import argparse
from typing import Any, Dict, List
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    set_seed,
    HfArgumentParser,
    AutoModelForSequenceClassification,
)
from tqdm import tqdm
import pyvene as pv 

from pyreft import (
    ReftConfig,
    ReftModel,
    get_reft_model,
    ReftDataCollator,
    ReftSupervisedDataset,
)
from csrf_trainer import CSRFTrainerForCausalLM
from csrf_router import CSRFRouter
from pyreft.interventions import LoreftIntervention

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Training script for CSRF model.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Path to pre-trained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--task_dataset_name",
        type=str,
        default="openbmb/UltraFeedback",
        help="Name of the training dataset."
    )
    parser.add_argument(
        "--task_dataset_split",
        type=str,
        default="train",
        help="Dataset split to use for training."
    )
    parser.add_argument(
        "--input_column",
        type=str,
        default="instruction",
        help="Name of the input column in the dataset."
    )
    parser.add_argument(
        "--output_column",
        type=str,
        default="completions",
        help="Name of the output column in the dataset."
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Maximum number of training samples. None for full dataset."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=float,
        default=1.0,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training."
    )
    parser.add_argument(
        "--num_subspaces",
        type=int,
        default=4,
        help="Number of subspaces for CSRF router."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass."
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Whether to run Alpaca evaluation after training"
    )
    return parser.parse_args()

def load_and_process_dataset(args):
    """Load and process the dataset."""
    dataset = load_dataset(args.task_dataset_name, split=args.task_dataset_split)
    
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(len(dataset), args.max_train_samples)))
    
    return dataset

def prepare_training_data(dataset, tokenizer, args):
    """
    Prepare training data so that only the 'response' tokens get supervised.
    Instruction tokens are masked out with label=-100.
    Also handles non-string fields by casting or converting them to strings.
    """
    def sanitize_text(val):
        """
        Convert arbitrary 'val' from the dataset into a safe string (possibly empty).
        If val is a list, we take the first element if it exists, else empty.
        """
        if val is None:
            return ""
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            if len(val) > 0:
                return val[0] if isinstance(val[0], str) else str(val[0])
            else:
                return ""
        return str(val)

    def process_example(example):
        instruction = sanitize_text(example.get(args.input_column))
        completion = sanitize_text(example.get(args.output_column))

        prompt_str = f"### Instruction:\n{instruction}\n\n### Response:\n"

        prompt_tokens = tokenizer(
            prompt_str,
            truncation=True,
            max_length=1024,
            add_special_tokens=False
        )
        response_tokens = tokenizer(
            completion,
            truncation=True,
            max_length=1024,
            add_special_tokens=False
        )

        input_ids = prompt_tokens["input_ids"] + response_tokens["input_ids"]
        attention_mask = [1] * len(input_ids)

        labels = [-100]*len(prompt_tokens["input_ids"]) + response_tokens["input_ids"]

        max_length = 2048
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    processed_dataset = dataset.map(
        process_example,
        remove_columns=dataset.column_names, 
        desc="Tokenizing dataset with instruction masking"
    )

    return processed_dataset


def generate_response(instruction, model, router, tokenizer, max_length=768):
    """Generate a response using the CSRF model and router."""
    prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    embeddings = model.model.get_input_embeddings()(inputs.input_ids)
    selected_embeddings = embeddings[:, 0, :]
    
    with torch.no_grad():
        gating_probs = router(selected_embeddings)
        active_subspaces = router.get_active_subspaces(gating_probs)
    
    with torch.no_grad():
        with model.interventions_active(active_subspaces):
            outputs = model.generate(
                base={
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_length": max_length,
                    "num_return_sequences": 1,
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.9,
                    "pad_token_id": tokenizer.eos_token_id,
                    "use_cache": True
                }
            )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("### Response:\n")[-1].strip()
    return response

def run_alpaca_eval(model, router, tokenizer, output_dir):
    """Run Alpaca evaluation on the trained model."""
    logger.info("Running Alpaca evaluation...")
    
    with open("gpt3_alpaca.json", "r") as f:
        eval_data = json.load(f)
    
    outputs = []
    for item in tqdm(eval_data):
        try:
            response = generate_response(item["instruction"], model, router, tokenizer)
            outputs.append({
                "dataset": item["dataset"],
                "instruction": item["instruction"], 
                "output": response,
                "generator": "csrf_model"
            })
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            continue

        logger.info(f"Response: {response}")
    
    output_file = "alpaca_eval_outputs_llama2_creft.json"
    with open(output_file, "w") as f:
        json.dump(outputs, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_file}")

def main():
    """Main training function."""
    args = parse_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device
    )
    base_model.config.use_cache = False
    
    for param in base_model.parameters():
        param.requires_grad = True
    
    reft_config = ReftConfig()
    
    model = get_reft_model(base_model, reft_config)
    model.set_device(device)
    model.train()
    
    for param in model.parameters():
        param.requires_grad = True
    
    dataset = load_and_process_dataset(args)
    processed_dataset = prepare_training_data(dataset, tokenizer, args)
    
    router = CSRFRouter(
        input_dim=base_model.config.hidden_size,
        num_subspaces=args.num_subspaces
    ).to(device).to(torch.float32)
    router.train()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=1,
        evaluation_strategy="no",
        save_strategy="no",
        report_to=["wandb"],
        remove_unused_columns=True,
        dataloader_num_workers=4,
        logging_first_step=True,
        max_grad_norm=1.0
    )
    
    trainer = CSRFTrainerForCausalLM(
        router=router,
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            return_tensors="pt",
            pad_to_multiple_of=8,
        ),
    )
    
    logger.info("Starting training...")
    trainer.model.train()
    trainer.router.train()
    
    trainer.train()
    
    trainer.save_model(args.output_dir)
    torch.save(router.state_dict(), os.path.join(args.output_dir, "router.pt"))
    model.save_intervention(
        os.path.join(args.output_dir, "intervenable_model"),
        include_model=True
    )
    
    logger.info(f"Model and components saved to {args.output_dir}")


    
    if args.run_eval:
        model.eval()
        router.eval()
        run_alpaca_eval(model, router, tokenizer, args.output_dir)

if __name__ == "__main__":
    setup_logging()
    main()

