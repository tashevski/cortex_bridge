#!/usr/bin/env python3
"""
Advanced Gemma Fine-tuner with LoRA support
===========================================

This script provides advanced fine-tuning capabilities including:
- LoRA (Low-Rank Adaptation) for efficient training
- Better conversation preprocessing
- Validation and evaluation metrics
- Multiple training strategies
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import subprocess
from datetime import datetime
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import re

from config import FineTuningConfig, get_config

class AdvancedGemmaFineTuner:
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the advanced Gemma fine-tuner.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.data_path = Path(config.data_path)
        self.output_dir = Path(config.output_dir)
        self.version_suffix = config.version_suffix or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = self.output_dir / f'fine_tuning_{self.version_suffix}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.tokenizer = None
        self.model = None
        
        # Log configuration
        self.logger.info(f"Initialized fine-tuner with config: {config}")
        
    def load_data(self) -> Dict[str, Any]:
        """Load and validate the training data."""
        self.logger.info(f"Loading data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.logger.info(f"Loaded {len(data)} conversations")
        return data
        
    def clean_conversation_text(self, text: str) -> str:
        """Clean and normalize conversation text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Ensure proper speaker formatting
        text = re.sub(r'Speaker_([A-Z]):', r'Human:', text)
        text = re.sub(r'Gemma:', r'Assistant:', text)
        
        return text
        
    def filter_conversations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter conversations based on configuration criteria."""
        filtered_data = {}
        
        for conv_id, conv_data in data.items():
            # Check message count
            message_count = conv_data.get('message_count', 0)
            if message_count < self.config.min_message_count or message_count > self.config.max_message_count:
                continue
                
            # Check feedback if filtering is enabled
            if self.config.filter_by_feedback:
                feedback = conv_data.get('feedback_helpful', 'False')
                if isinstance(feedback, str):
                    is_helpful = feedback.lower() == 'true'
                else:
                    is_helpful = bool(feedback)
                    
                if not is_helpful and not self.config.include_negative_examples:
                    continue
                    
            # Check if conversation has content
            full_text = conv_data.get('full_text', '').strip()
            if not full_text:
                continue
                
            filtered_data[conv_id] = conv_data
            
        self.logger.info(f"Filtered to {len(filtered_data)} conversations")
        return filtered_data
        
    def preprocess_conversations(self, data: Dict[str, Any]) -> Tuple[List[str], List[bool]]:
        """
        Preprocess conversations into training format.
        
        Returns:
            texts: List of formatted conversation texts
            labels: List of feedback labels
        """
        texts = []
        labels = []
        
        # Filter conversations first
        filtered_data = self.filter_conversations(data)
        
        for conv_id, conv_data in filtered_data.items():
            feedback = conv_data.get('feedback_helpful', 'False')
            
            # Normalize feedback to boolean
            if isinstance(feedback, str):
                is_helpful = feedback.lower() == 'true'
            else:
                is_helpful = bool(feedback)
                
            full_text = self.clean_conversation_text(conv_data.get('full_text', ''))
            
            if full_text:
                # Format as instruction-following conversation
                formatted_text = self.format_conversation_for_training(full_text, is_helpful)
                texts.append(formatted_text)
                labels.append(is_helpful)
                
        self.logger.info(f"Preprocessed {len(texts)} conversations")
        self.logger.info(f"Helpful conversations: {sum(labels)}")
        self.logger.info(f"Not helpful conversations: {len(labels) - sum(labels)}")
        
        return texts, labels
        
    def format_conversation_for_training(self, conversation: str, is_helpful: bool) -> str:
        """Format conversation for instruction-following training."""
        # Split conversation into turns
        turns = conversation.split('\n')
        formatted_turns = []
        
        for turn in turns:
            turn = turn.strip()
            if not turn:
                continue
                
            if turn.startswith('Human:') or turn.startswith('Speaker_'):
                # User turn
                user_msg = turn.split(':', 1)[1].strip()
                formatted_turns.append(f"<start_of_turn>user\n{user_msg}<end_of_turn>")
            elif turn.startswith('Assistant:') or turn.startswith('Gemma:'):
                # Assistant turn
                assistant_msg = turn.split(':', 1)[1].strip()
                formatted_turns.append(f"<start_of_turn>model\n{assistant_msg}<end_of_turn>")
                
        # Join turns
        formatted_conversation = '\n'.join(formatted_turns)
        
        # Add quality indicator if using negative examples
        if self.config.include_negative_examples and not is_helpful:
            formatted_conversation = f"# This is an example of a less helpful response\n{formatted_conversation}"
            
        return formatted_conversation
        
    def create_dataset(self, texts: List[str], labels: List[bool]) -> DatasetDict:
        """Create train/validation dataset split."""
        # Combine texts and labels
        data_dict = {
            "text": texts,
            "is_helpful": labels
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        # Split into train/validation
        if self.config.validation_split > 0:
            split_dataset = dataset.train_test_split(test_size=self.config.validation_split, seed=42)
            return DatasetDict({
                "train": split_dataset["train"],
                "validation": split_dataset["test"]
            })
        else:
            return DatasetDict({"train": dataset})
            
    def load_model_and_tokenizer(self):
        """Load the base Gemma model and tokenizer."""
        self.logger.info(f"Loading model and tokenizer: {self.config.base_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.config.use_fp16 and torch.cuda.is_available() else torch.float32,
            }
            
            if torch.cuda.is_available() and self.config.use_gpu:
                model_kwargs["device_map"] = "auto"
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **model_kwargs
            )
            
            # Add padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Setup LoRA if enabled
            if self.config.use_lora:
                self.setup_lora()
                
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def setup_lora(self):
        """Setup LoRA for efficient fine-tuning."""
        self.logger.info("Setting up LoRA for efficient fine-tuning")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def tokenize_function(self, examples):
        """Tokenize the training examples."""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
        
    def fine_tune_model(self, dataset: DatasetDict) -> str:
        """Fine-tune the Gemma model."""
        self.logger.info("Starting fine-tuning process")
        
        # Tokenize datasets
        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Training arguments
        output_path = self.output_dir / f"gemma-finetuned-{self.version_suffix}"
        
        training_args = TrainingArguments(
            output_dir=str(output_path),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.save_steps if "validation" in tokenized_datasets else None,
            save_total_limit=2,
            prediction_loss_only=True,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=self.config.dataloader_num_workers,
            fp16=self.config.use_fp16 and torch.cuda.is_available(),
            evaluation_strategy="steps" if "validation" in tokenized_datasets else "no",
            load_best_model_at_end=True if "validation" in tokenized_datasets else False,
            metric_for_best_model="eval_loss" if "validation" in tokenized_datasets else None,
            greater_is_better=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Callbacks
        callbacks = []
        if "validation" in tokenized_datasets:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets.get("validation"),
            callbacks=callbacks,
        )
        
        # Train
        trainer.train()
        
        # Save the model
        trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save training info
        training_info = {
            "version": self.version_suffix,
            "base_model": self.config.base_model,
            "training_samples": len(tokenized_datasets["train"]),
            "validation_samples": len(tokenized_datasets.get("validation", [])),
            "config": self.config.__dict__,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_path / "training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        self.logger.info(f"Model saved to: {output_path}")
        return str(output_path)
        
    def create_ollama_modelfile(self, model_path: str) -> str:
        """Create a Modelfile for Ollama."""
        modelfile_path = Path(model_path) / "Modelfile"
        
        modelfile_content = f"""# Gemma Fine-tuned Model - Version {self.version_suffix}
# Fine-tuned on conversation data with feedback
# Base model: {self.config.base_model}

FROM {model_path}

TEMPLATE \"\"\"{{{{ if .System }}}}<start_of_turn>system
{{{{ .System }}}}<end_of_turn>
{{{{ end }}}}{{{{ if .Prompt }}}}<start_of_turn>user
{{{{ .Prompt }}}}<end_of_turn>
<start_of_turn>model
{{{{ end }}}}{{{{ .Response }}}}<end_of_turn>
\"\"\"

PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
PARAMETER temperature 0.7
PARAMETER top_p 0.9

SYSTEM \"\"\"You are a helpful AI assistant that has been fine-tuned to provide better responses based on user feedback. You should be helpful, accurate, and engaging in your responses.\"\"\"
"""
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
            
        self.logger.info(f"Created Modelfile: {modelfile_path}")
        return str(modelfile_path)
        
    def create_ollama_model(self, model_path: str, model_name: str = None) -> str:
        """Create an Ollama model from the fine-tuned model."""
        if model_name is None:
            model_name = self.config.ollama_model_name or f"gemma-finetuned-{self.version_suffix}"
            
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
        
    def run_fine_tuning(self, model_name: str = None) -> Tuple[str, Optional[str]]:
        """Run the complete fine-tuning pipeline."""
        try:
            # Load and preprocess data
            data = self.load_data()
            texts, labels = self.preprocess_conversations(data)
            
            if not texts:
                raise ValueError("No training examples found. Check your data and filtering settings.")
                
            # Create dataset
            dataset = self.create_dataset(texts, labels)
            
            # Load model
            self.load_model_and_tokenizer()
            
            # Fine-tune
            model_path = self.fine_tune_model(dataset)
            
            # Create Ollama model if requested
            ollama_model_name = None
            if self.config.create_ollama_model:
                ollama_model_name = self.create_ollama_model(model_path, model_name)
                
            return model_path, ollama_model_name
                
        except Exception as e:
            self.logger.error(f"Fine-tuning failed: {e}")
            raise


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced fine-tuning for Gemma models")
    parser.add_argument("--config", default="default", 
                       help="Configuration preset to use")
    parser.add_argument("--data", help="Override data path")
    parser.add_argument("--model", help="Override base model")
    parser.add_argument("--name", help="Ollama model name")
    parser.add_argument("--no-ollama", action="store_true",
                       help="Skip creating Ollama model")
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_config(args.config)
    
    # Override settings from command line
    if args.data:
        config.data_path = args.data
    if args.model:
        config.base_model = args.model
    if args.no_ollama:
        config.create_ollama_model = False
    
    # Create fine-tuner
    fine_tuner = AdvancedGemmaFineTuner(config)
    
    # Run fine-tuning
    model_path, ollama_name = fine_tuner.run_fine_tuning(model_name=args.name)
    
    print(f"\n✅ Fine-tuning completed!")
    print(f"📁 Model saved to: {model_path}")
    if ollama_name:
        print(f"🦙 Ollama model created: {ollama_name}")
        print(f"🚀 Run with: ollama run {ollama_name}")


if __name__ == "__main__":
    main()