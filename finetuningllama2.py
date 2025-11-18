# finetune_llama2.py
import os
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from huggingface_hub import login

def main():
    # Authenticate with Hugging Face
    login(token="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")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("No GPU available, using CPU")
        return
    
    # Load dataset
    #dataset = load_from_disk("libertas_hip_dataset")
    #print(f"Loaded dataset with {len(dataset)} examples")
    
    # Model name - Llama 2 7B Chat
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    
    # More aggressive quantization configuration for 4GB GPU
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
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # More conservative LoRA configuration for memory efficiency
    lora_config = LoraConfig(
        r=8,                      # Lower rank
        lora_alpha=16,            # Smaller alpha
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[          # Focus on attention layers only
            "q_proj", "k_proj", "v_proj", "o_proj"
        ],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Formatting function for chat models
    def format_prompt(example):
        messages = [
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
    
    #formatted_dataset = dataset.map(format_prompt)

    def format_prompt(example):
    # For QA format
        if example['input']:
            prompt = (f"### Instruction: {example['instruction']}\n"
                    f"### Input: {example['input']}\n"
                    f"### Response: {example['output']}")
        else:
            prompt = (f"### Instruction: {example['instruction']}\n"
                    f"### Response: {example['output']}")
        return {"text": prompt}

    # Then in your main():
    dataset = load_from_disk("libertas_hip_qa_dataset")  # Load our preprocessed data
    formatted_dataset = dataset.map(format_prompt)
        
    # Tokenization
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    
    tokenized_dataset = formatted_dataset.map(tokenize_function, batched=True)
    
    # Output directory
    output_dir = "llama2_7b_libertas_finetuned"
    
    # Conservative training arguments for 4GB GPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,                  # Fewer epochs
        per_device_train_batch_size=1,       # Small batch size
        gradient_accumulation_steps=8,       # Higher accumulation
        learning_rate=1e-4,                  # Lower learning rate
        weight_decay=0.01,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,         # Important for chat templates
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Test inference
    print("Testing inference...")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    messages = [{"role": "user", "content": "Who are you?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    result = pipe(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
    print(result[0]['generated_text'])

if __name__ == "__main__":
    main()