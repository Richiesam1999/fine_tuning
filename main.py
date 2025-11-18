import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load configuration
    peft_model_path = "libertas_hip_finetuned_model"
    config = PeftConfig.from_pretrained(peft_model_path)
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for memory efficiency
        load_in_4bit=True,         # Load in 4-bit quantization
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(model, peft_model_path)
    
    # Testing function
    def generate_response(prompt, max_length=512):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Test with a sample prompt
    test_prompt = "Background Treatment of tibia (upper third and diaphysis) fracture"
    response = generate_response(test_prompt)
    
    print("=" * 50)
    print(f"Prompt: {test_prompt}")
    print("=" * 50)
    print(f"Response: {response}")
    print("=" * 50)

if __name__ == "__main__":
    main()