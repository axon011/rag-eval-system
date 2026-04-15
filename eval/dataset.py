import json
import os
from typing import List, Dict, Any
from datetime import datetime


class EvalDataset:
    def __init__(self, dataset_path: str = None):
        self.dataset_path = dataset_path or os.path.join(
            os.path.dirname(__file__), 
            "dataset.json"
        )
        self.dataset = []
        
        if os.path.exists(self.dataset_path):
            self.load()
        else:
            self.dataset = self._get_default_dataset()
    
    def _get_default_dataset(self) -> List[Dict[str, str]]:
        # Try loading from dataset.json first
        json_path = os.path.join(os.path.dirname(__file__), "dataset.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)

        # Fallback — should never reach here if dataset.json exists
        return [
            {
                "question": "What is LangGraph and what problem does it solve?",
                "answer": "LangGraph is a framework for building stateful, multi-agent applications with LLMs.",
                "context": "LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows."
            }
        ]
    
    def load(self):
        with open(self.dataset_path, 'r') as f:
            self.dataset = json.load(f)
    
    def save(self):
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)
        with open(self.dataset_path, 'w') as f:
            json.dump(self.dataset, f, indent=2)
    
    def get_dataset(self) -> List[Dict[str, str]]:
        return self.dataset
    
    def add_question(self, question: str, answer: str, context: str):
        self.dataset.append({
            "question": question,
            "answer": answer,
            "context": context
        })
    
    def get_questions(self) -> List[str]:
        return [item["question"] for item in self.dataset]
    
    def get_ground_truth_answers(self) -> List[str]:
        return [item["answer"] for item in self.dataset]
    
    def get_contexts(self) -> List[str]:
        return [item["context"] for item in self.dataset]
