import re
from datasets import Dataset

def split_into_sections(text):
    """Split the document into logical sections"""
    # Split by all caps headings followed by newline
    sections = re.split(r'\n([A-Z][A-Z\s®©°-]+[A-Z])\n', text)
    
    # The first element is usually content before first heading
    sections = [sections[0]] + list(zip(sections[1::2], sections[2::2]))
    
    # Clean sections
    cleaned_sections = []
    for section in sections:
        if isinstance(section, tuple):
            heading, content = section
            cleaned_sections.append({
                'heading': heading.strip(),
                'content': content.strip()
            })
        else:
            if section.strip():
                cleaned_sections.append({
                    'heading': 'INTRODUCTION',
                    'content': section.strip()
                })
    return cleaned_sections

def create_qa_pairs(sections):
    """Generate question-answer pairs from document sections"""
    qa_pairs = []
    
    # Create general questions about each section
    for section in sections:
        heading = section['heading']
        content = section['content']
        
        # Basic questions
        q1 = f"What is the {heading} section about?"
        q2 = f"Summarize the {heading} section"
        q3 = f"What information does the {heading} section provide?"
        
        qa_pairs.extend([
            {'instruction': q1, 'input': '', 'output': content},
            {'instruction': q2, 'input': '', 'output': content},
            {'instruction': q3, 'input': '', 'output': content},
        ])
        
        # For technical specifications, create more specific questions
        if 'SPECIFICATIONS' in heading or 'DETAILS' in heading:
            # Extract key-value pairs
            items = re.findall(r'•\s*(.*?)\n', content)
            for item in items:
                if ':' in item:
                    key, value = item.split(':', 1)
                    q = f"What is the {key.strip()} of the {heading.replace('MATERIAL SPECIFICATIONS', '').strip()}?"
                    qa_pairs.append({
                        'instruction': q, 
                        'input': '', 
                        'output': value.strip()
                    })
    
    return qa_pairs

def preprocess_data(text_file_path):
    # Load your text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sections
    sections = split_into_sections(text)
    
    # Create QA pairs
    qa_pairs = create_qa_pairs(sections)
    
    # Convert to HuggingFace dataset
    dataset = Dataset.from_dict({
        'instruction': [pair['instruction'] for pair in qa_pairs],
        'input': [pair['input'] for pair in qa_pairs],
        'output': [pair['output'] for pair in qa_pairs]
    })
    
    # Save the processed dataset
    dataset.save_to_disk("libertas_hip_qa_dataset")
    return dataset

# Usage
preprocess_data("Libertas hip system overview_extracted.txt")