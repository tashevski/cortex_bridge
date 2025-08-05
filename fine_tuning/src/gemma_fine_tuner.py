#!/usr/bin/env python3
"""
Gemma Model Fine-tuning Script for Ollama
=========================================

This script fine-tunes Gemma models using conversation data with feedback ratings.
It creates new model versions that can be pulled with Ollama without overwriting existing ones.

Data format expected:
{
    "conversation_N": {
        "feedback_helpful": "True" | "False" | true | false,
        "full_text": "Speaker_A: ...\nGemma: ...\n...",
        "session_id": "session_...",
        "timestamp": "...",
        "message_count": N
    }
}
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import subprocess
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np

class GemmaFineTuner:
    def __init__(self, 
                 data_path: str = "data/training_data.json",
                 base_model: str = "google/gemma-2b-it",
                 output_dir: str = "models",
                 version_suffix: str = None):
        """
        Initialize the Gemma fine-tuner.
        
        Args:
            data_path: Path to the training data JSON file
            base_model: Base Gemma model name/path
            output_dir: Directory to save fine-tuned models
            version_suffix: Version suffix for the model (auto-generated if None)
        """
        self.data_path = Path(data_path)
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.version_suffix = version_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'fine_tuning_{self.version_suffix}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.tokenizer = None
        self.model = None
        
    def load_data(self) -> Dict[str, Any]:
        """Load and validate the training data."""
        self.logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.logger.info(f"Loaded {len(data)} conversations")
        return data
        
    def preprocess_conversations(self, data: Dict[str, Any]) -> Tuple[List[str], List[bool]]:
        """
        Preprocess conversations into training format.
        
        Returns:
            texts: List of conversation texts
            labels: List of feedback labels (True for helpful, False for not helpful)
        """
        texts = []
        labels = []
        
        for conv_id, conv_data in data.items():
            feedback = conv_data.get('feedback_helpful', 'False')
            
            # Normalize feedback to boolean
            if isinstance(feedback, str):
                is_helpful = feedback.lower() == 'true'
            else:
                is_helpful = bool(feedback)
                
            full_text = conv_data.get('full_text', '')
            
            if full_text.strip():
                texts.append(full_text)
                labels.append(is_helpful)
                
        self.logger.info(f"Preprocessed {len(texts)} conversations")
        self.logger.info(f"Helpful conversations: {sum(labels)}")
        self.logger.info(f"Not helpful conversations: {len(labels) - sum(labels)}")
        
        return texts, labels
        
    def create_training_examples(self, texts: List[str], labels: List[bool]) -> List[str]:
        """
        Create training examples in instruction format.
        
        For helpful conversations, we keep them as positive examples.
        For unhelpful conversations, we modify them to show better responses.
        """
        training_examples = []
        
        for text, is_helpful in zip(texts, labels):
            if is_helpful:
                # Keep helpful conversations as positive examples
                training_examples.append(text)
            else:
                # For unhelpful conversations, we can either:
                # 1. Skip them (current approach)
                # 2. Create negative examples with corrective feedback
                # 3. Modify them to show better responses
                
                # For now, we'll skip unhelpful conversations
                # but log them for potential manual review
                self.logger.debug(f"Skipping unhelpful conversation: {text[:100]}...")
                
        self.logger.info(f"Created {len(training_examples)} training examples")
        return training_examples
        
    def load_model_and_tokenizer(self):
        """Load the base Gemma model and tokenizer."""
        self.logger.info(f"Loading model and tokenizer: {self.base_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def tokenize_function(self, examples):
        """Tokenize the training examples."""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
        
    def fine_tune_model(self, training_examples: List[str]):
        """Fine-tune the Gemma model."""
        self.logger.info("Starting fine-tuning process")
        
        # Create dataset
        dataset_dict = {"text": training_examples}
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/gemma-finetuned-{self.version_suffix}",
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=100,
            learning_rate=5e-5,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset,
        )
        
        # Train
        trainer.train()
        
        # Save the model
        final_model_path = f"{self.output_dir}/gemma-finetuned-{self.version_suffix}"
        trainer.save_model(final_model_path)
        self.tokenizer.save_pretrained(final_model_path)
        
        self.logger.info(f"Model saved to: {final_model_path}")
        return final_model_path
        
    def create_ollama_modelfile(self, model_path: str) -> str:
        """Create a Modelfile for Ollama."""
        modelfile_path = f"{model_path}/Modelfile"
        
        modelfile_content = f"""# Gemma Fine-tuned Model - Version {self.version_suffix}
# Fine-tuned on conversation data with feedback

FROM {model_path}

TEMPLATE \"\"\"{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
<start_of_turn>model
{{ end }}{{ .Response }}<end_of_turn>
\"\"\"

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"

SYSTEM \"\"\"You are a helpful AI assistant. You have been fine-tuned on conversation data to provide more helpful and relevant responses.\"\"\"
"""
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
            
        self.logger.info(f"Created Modelfile: {modelfile_path}")
        return modelfile_path
        
    def create_ollama_model(self, model_path: str, model_name: str = None) -> str:
        """Create an Ollama model from the fine-tuned model."""
        if model_name is None:
            model_name = f"gemma-finetuned-{self.version_suffix}"
            
        # Create Modelfile
        modelfile_path = self.create_ollama_modelfile(model_path)
        
        # Create Ollama model
        cmd = f"ollama create {model_name} -f {modelfile_path}"
        
        self.logger.info(f"Creating Ollama model: {cmd}")
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                self.logger.info(f"Successfully created Ollama model: {model_name}")
                self.logger.info(f"You can now use: ollama run {model_name}")
            else:
                self.logger.error(f"Failed to create Ollama model: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error creating Ollama model: {e}")
            
        return model_name
        
    def run_fine_tuning(self, create_ollama: bool = True, model_name: str = None):
        """Run the complete fine-tuning pipeline."""
        try:
            # Load and preprocess data
            data = self.load_data()
            texts, labels = self.preprocess_conversations(data)
            training_examples = self.create_training_examples(texts, labels)
            
            if not training_examples:
                raise ValueError("No training examples found. Check your data and feedback labels.")
                
            # Load model
            self.load_model_and_tokenizer()
            
            # Fine-tune
            model_path = self.fine_tune_model(training_examples)
            
            # Create Ollama model if requested
            if create_ollama:
                ollama_model_name = self.create_ollama_model(model_path, model_name)
                return model_path, ollama_model_name
            else:
                return model_path, None
                
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Gemma models for Ollama")
    parser.add_argument("--data", default="data/training_data.json", 
                       help="Path to training data JSON file")
    parser.add_argument("--model", default="google/gemma-2b-it",
                       help="Base Gemma model name")
    parser.add_argument("--output", default="models",
                       help="Output directory for fine-tuned models")
    parser.add_argument("--version", default=None,
                       help="Version suffix for the model")
    parser.add_argument("--name", default=None,
                       help="Ollama model name")
    parser.add_argument("--no-ollama", action="store_true",
                       help="Skip creating Ollama model")
    
    args = parser.parse_args()
    
    # Create fine-tuner
    fine_tuner = GemmaFineTuner(
        data_path=args.data,
        base_model=args.model,
        output_dir=args.output,
        version_suffix=args.version
    )
    
    # Run fine-tuning
    model_path, ollama_name = fine_tuner.run_fine_tuning(
        create_ollama=not args.no_ollama,
        model_name=args.name
    )
    
    print(f"\n✅ Fine-tuning completed!")
    print(f"📁 Model saved to: {model_path}")
    if ollama_name:
        print(f"🦙 Ollama model created: {ollama_name}")
        print(f"🚀 Run with: ollama run {ollama_name}")


if __name__ == "__main__":
    main()