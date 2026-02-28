import os
import json
from datetime import datetime
from typing import Dict, Any, List
from tqdm import tqdm

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from datasets import Dataset

from eval.dataset import EvalDataset
from app.core.pipeline import RAGPipeline


class EvalRunner:
    def __init__(
        self,
        results_dir: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        top_k: int = None,
        retrieval_mode: str = None
    ):
        self.results_dir = results_dir or os.path.join(
            os.path.dirname(__file__),
            "results"
        )
        
        self.config = {
            "chunk_size": chunk_size or int(os.getenv("CHUNK_SIZE", "512")),
            "chunk_overlap": chunk_overlap or int(os.getenv("CHUNK_OVERLAP", "64")),
            "top_k": top_k or int(os.getenv("TOP_K", "5")),
            "retrieval_mode": retrieval_mode or os.getenv("RETRIEVAL_MODE", "hybrid"),
            "embed_model": os.getenv("EMBED_MODEL", "nomic-embed-text"),
            "llm_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "eval_date": datetime.now().isoformat()
        }
        
        os.makedirs(self.results_dir, exist_ok=True)
    
    def run_evaluation(self) -> Dict[str, Any]:
        print("Loading evaluation dataset...")
        dataset = EvalDataset()
        eval_data = dataset.get_dataset()
        
        print(f"Running evaluation on {len(eval_data)} questions...")
        
        pipeline = RAGPipeline()
        
        questions = []
        answers = []
        contexts = []
        
        for item in tqdm(eval_data, desc="Processing questions"):
            question = item["question"]
            ground_truth_answer = item["answer"]
            
            result = pipeline.query(question, rewrite_query=True)
            
            retrieved_contexts = [source["text"] for source in result["sources"]]
            generated_answer = result["answer"]
            
            questions.append(question)
            answers.append(generated_answer)
            contexts.append(retrieved_contexts)
        
        print("Computing RAGAs metrics...")
        
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": [item["answer"] for item in eval_data]
        })
        
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
        
        eval_results = evaluate(
            eval_dataset,
            metrics=metrics
        )
        
        results_dict = {
            "config": self.config,
            "metrics": {
                "faithfulness": eval_results["faithfulness"],
                "answer_relevancy": eval_results["answer_relevancy"],
                "context_recall": eval_results["context_recall"],
                "context_precision": eval_results["context_precision"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        results_file = os.path.join(
            self.results_dir,
            f"run_{datetime.now().strftime('%Y-%m-%d')}.json"
        )
        
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        return results_dict
    
    def get_latest_results(self) -> Dict[str, Any]:
        if not os.path.exists(self.results_dir):
            return None
        
        files = [f for f in os.listdir(self.results_dir) if f.endswith('.json')]
        
        if not files:
            return None
        
        latest_file = max(files)
        latest_path = os.path.join(self.results_dir, latest_file)
        
        with open(latest_path, 'r') as f:
            return json.load(f)


if __name__ == "__main__":
    runner = EvalRunner()
    results = runner.run_evaluation()
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness: {results['metrics']['faithfulness']:.3f}")
    print(f"Answer Relevancy: {results['metrics']['answer_relevancy']:.3f}")
    print(f"Context Recall: {results['metrics']['context_recall']:.3f}")
    print(f"Context Precision: {results['metrics']['context_precision']:.3f}")
    print("="*50)
