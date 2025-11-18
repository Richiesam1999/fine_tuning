import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def main():
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
    
    # Load dataset
    try:
        dataset = load_from_disk("output_datasets/hf_dataset")
        print(f"Loaded dataset with {len(dataset)} examples")
        train_dataset = dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Define model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    # Quantization configuration
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    print(f"Loading tokenizer from {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model from {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )
    
    print("Model loaded successfully")
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ],
    )
    
    # Get the PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Preprocess and tokenize dataset with proper padding
    def preprocess_function(examples):
        # Format prompt
        prompts = []
        for i in range(len(examples['instruction'])):
            prompt = f"### Instruction: {examples['instruction'][i]}\n\n"
            if examples['input'][i]:
                prompt += f"{examples['input'][i]}\n\n"
            prompt += f"### Response: {examples['output'][i]}"
            prompts.append(prompt)
        
        # Tokenize with padding and truncation
        return tokenizer(
            prompts,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Output directory
    output_dir = "libertas_hip_finetuned_model"
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()