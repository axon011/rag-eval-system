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
        return [
            {
                "question": "What is the main topic of this document?",
                "answer": "The main topic is [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What are the key findings presented?",
                "answer": "The key findings are [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What methodology was used?",
                "answer": "The methodology used was [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What are the main conclusions?",
                "answer": "The main conclusions are [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What data was analyzed in this study?",
                "answer": "The data analyzed was [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What are the limitations mentioned?",
                "answer": "The limitations mentioned are [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What are the recommendations provided?",
                "answer": "The recommendations provided are [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What previous research is referenced?",
                "answer": "Previous research referenced includes [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What are the implications of this work?",
                "answer": "The implications of this work are [extract from your document]",
                "context": "The relevant context from the document"
            },
            {
                "question": "What future work is suggested?",
                "answer": "Future work suggested includes [extract from your document]",
                "context": "The relevant context from the document"
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
