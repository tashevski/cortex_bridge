#!/usr/bin/env python3
"""Minimal Gemma fine-tuner with LoRA and Ollama integration"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

class MinimalGemmaFineTuner:
    def __init__(self, data_path="data/training_data.json", model="google/gemma-2b-it", version=None):
        self.data_path = Path(data_path)
        self.model_name = model
        self.version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"models/gemma-ft-{self.version}")
        
    def load_and_filter_data(self):
        """Load conversations and keep only helpful ones"""
        with open(self.data_path) as f:
            data = json.load(f)
        
        # Filter for helpful conversations only
        helpful_convs = []
        for conv in data.values():
            feedback = str(conv.get('feedback_helpful', 'False')).lower()
            if feedback == 'true' and conv.get('full_text', '').strip():
                # Format as instruction-response pairs
                text = conv['full_text'].replace('Speaker_A:', 'Human:').replace('Gemma:', 'Assistant:')
                helpful_convs.append(text)
        
        print(f"Using {len(helpful_convs)} helpful conversations")
        return helpful_convs
        
    def setup_model(self):
        """Load model with LoRA"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Add LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        
    def train(self, texts):
        """Fine-tune the model"""
        # Tokenize
        def tokenize(examples):
            tokens = self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
            
        dataset = Dataset.from_dict({"text": texts}).map(tokenize, batched=True)
        
        # Train
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir=str(self.output_dir),
                num_train_epochs=3,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                save_steps=500,
                logging_steps=50,
                fp16=torch.cuda.is_available()
            ),
            train_dataset=dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
    def create_ollama_model(self, name=None):
        """Create Ollama model"""
        model_name = name or f"gemma-ft-{self.version}"
        
        # Create Modelfile
        modelfile = self.output_dir / "Modelfile"
        modelfile.write_text(f"""FROM {self.output_dir}
TEMPLATE \"\"\"{{{{ if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{ end }}}}{{{{ if .Prompt }}}}<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
{{{{ end }}}}{{{{ .Response }}}}<end_of_turn>
\"\"\"
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
""")
        
        # Create model
        result = subprocess.run(f"ollama create {model_name} -f {modelfile}", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Created Ollama model: {model_name}")
            return model_name
        else:
            print(f"❌ Failed to create Ollama model: {result.stderr}")
            return None
    
    def run(self, ollama_name=None):
        """Complete fine-tuning pipeline"""
        texts = self.load_and_filter_data()
        if not texts:
            raise ValueError("No helpful conversations found")
            
        self.setup_model()
        self.train(texts)
        return self.create_ollama_model(ollama_name)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/training_data.json")
    parser.add_argument("--model", default="google/gemma-2b-it") 
    parser.add_argument("--name", help="Ollama model name")
    args = parser.parse_args()
    
    tuner = MinimalGemmaFineTuner(args.data, args.model)
    model_name = tuner.run(args.name)
    
    if model_name:
        print(f"🚀 Run with: ollama run {model_name}")

if __name__ == "__main__":
    main()