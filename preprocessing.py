import os
import re
import json
from datasets import Dataset

def process_single_text_file(file_path):
    """Process a single text file into training examples."""
    examples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into reasonable chunks (paragraphs or sections)
    chunks = split_into_chunks(content)
    
    for chunk in chunks:
        # Create instruction-response pairs
        instruction = "Provide information about the Libertas Hip System based on the following text."
        response = chunk
        
        examples.append({
            "instruction": instruction,
            "input": "",
            "output": response
        })
    
    return examples

def split_into_chunks(text, max_chunk_size=512):
    """Split text into chunks of reasonable size."""
    # Split by section headers or paragraphs
    sections = re.split(r'\n\s*\n|\n#+\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for section in sections:
        if not section.strip():
            continue
            
        if len(current_chunk) + len(section) < max_chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += section.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = section.strip()
    
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def create_dataset(examples):
    """Convert examples to HuggingFace dataset."""
    return Dataset.from_dict({
        "instruction": [ex["instruction"] for ex in examples],
        "input": [ex["input"] for ex in examples],
        "output": [ex["output"] for ex in examples]
    })

def main():
    # Process a single file instead of a directory
    file_path = "Libertas hip system overview_extracted.txt"  # Your text file
    
    # Process file
    examples = process_single_text_file(file_path)
    
    # Save to jsonl format
    output_jsonl = "training_data.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    # Create HuggingFace dataset
    dataset = create_dataset(examples)
    
    # Save dataset
    output_dataset = "libertas_hip_dataset"
    dataset.save_to_disk(output_dataset)
    
    print(f"Created dataset with {len(examples)} examples")
    print(f"JSONL file saved to: {output_jsonl}")
    print(f"Dataset saved to: {output_dataset}")

if __name__ == "__main__":
    main()