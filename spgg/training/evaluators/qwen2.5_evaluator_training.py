#!/usr/bin/env python3
"""
Evaluator Model Training Script.

Implements the structured text generation task for training the evaluator model
as specified in the paper. Only score spans are supervised during training.

Loss function: L = -sum_{t in T_score} log p_theta(y_t | x_i, y_{<t})
where T_score denotes the set of token indices corresponding to numeric score values.
"""

import os
import re
import gc
import numpy as np
import pandas as pd
import torch
import json
import warnings
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from sklearn.metrics import mean_squared_error, mean_absolute_error
from huggingface_hub import login
import wandb

import sys
sys.path.append('/content/drive/MyDrive/PGG/SPGG')

from training.evaluators.utils import (
    create_structured_prompt,
    extract_scores_from_response,
    load_summeval_dataset,
    clear_memory,
    setup_directories,
    mount_google_drive,
    get_device,
)

# Environment configuration
os.environ["HF_DATASETS_OFFLINE"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# HuggingFace authentication
hf_token = os.environ.get("HUGGINGFACE_TOKEN")
if hf_token:

    try:
        login(hf_token)
    except Exception:
        pass


# Model and training configuration
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
LORA_CONFIG = {
    "r": 8,
    "alpha": 16,
    "dropout": 0.05,
    "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
}
TRAINING_CONFIG = {
    "max_epochs": 20,
    "learning_rate": 8e-5,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "warmup_steps": 100,
    "early_stopping_patience": 3
}


def create_evaluator_dataset(examples, tokenizer):
    """
    Create dataset with structured text generation format.
    
    Implements selective supervision where only score spans are trained.
    All other tokens are masked with label -100.
    
    Args:
        examples: Batch of training examples.
        tokenizer: Tokenizer instance.
    
    Returns:
        Tokenized dataset with masked labels.
    """
    input_texts = []
    
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        summary = examples["summary"][i]
        r_relevance = float(examples["r_relevance"][i])
        r_coherence = float(examples["r_coherence"][i])
        r_consistency = float(examples["r_consistency"][i])
        r_fluency = float(examples["r_fluency"][i])
        
        full_text = create_structured_prompt(
            text, summary, r_relevance, r_coherence, r_consistency, r_fluency
        )
        input_texts.append(full_text)
    
    tokenized = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",
        max_length=2048,
        return_tensors=None
    )
    
    # Create labels with score-only supervision
    labels = []
    score_patterns = [
            r"Relevance: (\d+(?:\.\d+)?)",
            r"Coherence: (\d+(?:\.\d+)?)", 
            r"Consistency: (\d+(?:\.\d+)?)",
            r"Fluency: (\d+(?:\.\d+)?)"
        ]
        
    for i, input_ids in enumerate(tokenized["input_ids"]):
        label = [-100] * len(input_ids)
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        for pattern in score_patterns:
            match = re.search(pattern, text)
            if match:
                score_text = match.group(1)
                score_tokens = tokenizer.encode(score_text, add_special_tokens=False)
                
                for j in range(len(input_ids) - len(score_tokens) + 1):
                    if input_ids[j:j+len(score_tokens)] == score_tokens:
                        for k in range(len(score_tokens)):
                            label[j+k] = input_ids[j+k]
                        break
        
        labels.append(label)
    
    tokenized["labels"] = labels
    return tokenized


def train_evaluator_model(dataset, tokenizer, model_name):
    """
    Train evaluator model with QLoRA.
    
    Configuration:
    - 4-bit quantization with NF4
    - LoRA rank=8, alpha=16, dropout=0.05
    - AdamW optimizer with lr=8e-5
    - Early stopping with patience=3
    
    Args:
        dataset: Training and test datasets.
        tokenizer: Tokenizer instance.
        model_name: Base model name.
    
    Returns:
        Trained model and save path.
    """
    print(f"\n{'='*50}")
    print(f"Training Evaluator Model")
    print(f"{'='*50}")
    
    # Process datasets
    print("Creating structured training dataset...")
    train_dataset = dataset["train"].map(
        lambda examples: create_evaluator_dataset(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Processing training data"
    )
    
    test_dataset = dataset["test"].map(
        lambda examples: create_evaluator_dataset(examples, tokenizer),
        batched=True,
        remove_columns=dataset["test"].column_names,
        desc="Processing test data"
    )
    
    # Load model with 4-bit quantization
    print(f"Loading {model_name} with 4-bit quantization...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
        quantization_config=quantization_config,
        max_memory={0: "35GB"}
    )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["alpha"],
        lora_dropout=LORA_CONFIG["dropout"],
        target_modules=LORA_CONFIG["target_modules"],
        bias="none"
    )
    
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.train()
    
    print(f"LoRA config: r={peft_config.r}, alpha={peft_config.lora_alpha}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="/content/qwen2.5_evaluator_model",
        overwrite_output_dir=True,
        num_train_epochs=TRAINING_CONFIG["max_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        per_device_eval_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=True,
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        weight_decay=0.01,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="wandb",
        push_to_hub=False,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        run_name=f"qwen2.5-evaluator-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    print(f"Training config:")
    print(f"  Max epochs: {training_args.num_train_epochs}")
    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=TRAINING_CONFIG["early_stopping_patience"])]
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save model
    save_path = "/content/drive/MyDrive/qwen2.5_evaluator_model2"
    print(f"Saving model to: {save_path}")
    os.makedirs(save_path, exist_ok=True)
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    
    # Cleanup
    import shutil
    if os.path.exists("/content/qwen2.5_evaluator_model"):
        shutil.rmtree("/content/qwen2.5_evaluator_model")
    
    print(f"Model saved to: {save_path}")
    return model, save_path


def test_evaluator_model(model, tokenizer, test_samples):
    """
    Test the trained evaluator model.
    
    Generates reward vectors r = (r_relevance, r_coherence, r_consistency, r_fluency)
    and computes error metrics against ground truth.
    
    Args:
        model: Trained evaluator model.
        tokenizer: Tokenizer instance.
        test_samples: Test dataset samples.
    
    Returns:
        Tuple of (MSE, MAE) metrics.
    """
    print(f"\n{'='*50}")
    print("Testing Evaluator Model")
    print(f"{'='*50}")
    
    all_predictions = []
    all_true_scores = []
    
    for i, sample in enumerate(test_samples):
        try:
            print(f"\nTest sample {i+1}")
            print(f"Document: {sample['text'][:150]}...")
            print(f"Summary: {sample['summary'][:100]}...")
            
            prompt = create_structured_prompt(sample["text"], sample["summary"])
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    min_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,  
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"Output: {response}")
            
            predicted_rewards = extract_scores_from_response(response)
            
            true_rewards = {
                'relevance': float(sample["r_relevance"]),
                'coherence': float(sample["r_coherence"]),
                'consistency': float(sample["r_consistency"]),
                'fluency': float(sample["r_fluency"])
            }
            
            print("Score comparison:")
            for dim in ['relevance', 'coherence', 'consistency', 'fluency']:
                if dim in predicted_rewards:
                    pred = predicted_rewards[dim]
                true = true_rewards[dim]
                error = abs(pred - true)
                print(f"  {dim}: pred={pred:.2f}, true={true:.2f}, error={error:.2f}")
                all_predictions.append(pred)
                all_true_scores.append(true)
                    
        except Exception as e:
            print(f"Error processing sample {i+1}: {e}")
    
    if len(all_predictions) > 0:
        mse = mean_squared_error(all_true_scores, all_predictions)
        mae = mean_absolute_error(all_true_scores, all_predictions)
        print(f"\nOverall metrics:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        return mse, mae
    
        return None, None


def main():
    """Main training pipeline."""
    print("Evaluator Model Training")
    print("="*60)
    
    wandb.init(
        project="qwen2.5-evaluator-training",
        name=f"qwen2.5-evaluator-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "model_name": MODEL_NAME,
            "training_method": "QLoRA",
            **LORA_CONFIG,
            **TRAINING_CONFIG,
            "dataset": "SummEval"
        }
    )
    
    try:
        setup_directories()
        mount_google_drive()
        
        device = get_device()
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Load dataset
        summeval_dataset = load_summeval_dataset()
        dataset = summeval_dataset.shuffle(seed=42).train_test_split(test_size=0.1, seed=42)
        print(f"Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Baseline test
        print(f"\n{'='*60}")
        print("Pre-training baseline")
        print(f"{'='*60}")
        
        baseline_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        
        test_sample = dataset["test"].select(range(1))
        test_evaluator_model(baseline_model, tokenizer, test_sample)
        
        del baseline_model
        clear_memory()
        
        # Train model
        trained_model, model_path = train_evaluator_model(
            dataset=dataset,
            tokenizer=tokenizer,
            model_name=MODEL_NAME
        )
        
        # Post-training evaluation
        print(f"\n{'='*60}")
        print("Post-training evaluation")
        print(f"{'='*60}")
        
        test_samples = dataset["test"].select(range(5))
        mse, mae = test_evaluator_model(trained_model, tokenizer, test_samples)
        
        if mse is not None:
            wandb.log({"final_test_mse": mse, "final_test_mae": mae})
            
        print(f"\nTraining complete!")
        print(f"Model saved to: {model_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        wandb.log({"training_failed": True, "error": str(e)})
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main() 
