import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
import mlflow
from mlflow.tracking import MlflowClient


class MLflowTracker:
    def __init__(
        self,
        tracking_uri: str = None,
        experiment_name: str = "rag-evaluation"
    ):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", 
            "http://localhost:5000"
        )
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(self.tracking_uri)
        
        try:
            mlflow.set_experiment(self.experiment_name)
        except Exception:
            pass
    
    def log_experiment(
        self,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        run_name: Optional[str] = None
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            mlflow.log_param("eval_timestamp", datetime.now().isoformat())
            
            run_id = run.info.run_id
            
            print(f"Logged experiment to MLflow. Run ID: {run_id}")
            
            return run_id
    
    def log_params_only(
        self,
        config: Dict[str, Any],
        run_name: Optional[str] = None
    ) -> str:
        with mlflow.start_run(run_name=run_name) as run:
            for key, value in config.items():
                mlflow.log_param(key, value)
            
            mlflow.log_param("eval_timestamp", datetime.now().isoformat())
            
            return run.info.run_id
    
    def log_metrics_only(
        self,
        metrics: Dict[str, float],
        run_id: Optional[str] = None
    ):
        if run_id:
            client = MlflowClient()
            for key, value in metrics.items():
                client.log_metric(run_id, mlflow.entities.Metric(key, value, 0, 0))
        else:
            with mlflow.start_run() as run:
                for key, value in metrics.items():
                    mlflow.log_metric(key, value)
    
    def get_experiment_runs(self) -> list:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if not experiment:
            return []
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        
        return runs
    
    def compare_configs(
        self,
        param_name: str,
        metric_name: str = "faithfulness"
    ) -> Dict[str, Any]:
        runs = self.get_experiment_runs()
        
        if runs.empty:
            return {"message": "No runs found"}
        
        param_col = f"params.{param_name}"
        metric_col = f"metrics.{metric_name}"
        
        if param_col not in runs.columns or metric_col not in runs.columns:
            return {"message": f"Parameter '{param_name}' or metric '{metric_name}' not found"}
        
        comparison = runs[[param_col, metric_col]].sort_values(metric_col, ascending=False)
        
        return comparison.to_dict(orient="records")
    
    def get_best_run(self, metric: str = "faithfulness") -> Optional[Dict[str, Any]]:
        runs = self.get_experiment_runs()
        
        if runs.empty:
            return None
        
        metric_col = f"metrics.{metric}"
        
        if metric_col not in runs.columns:
            return None
        
        best_idx = runs[metric_col].idxmax()
        best_run = runs.loc[best_idx]
        
        return {
            "run_id": best_run["run_id"],
            f"{metric}": best_run[metric_col],
            "params": {k.replace("params.", ""): best_run[k] 
                      for k in runs.columns if k.startswith("params.")}
        }


if __name__ == "__main__":
    tracker = MLflowTracker()
    
    test_config = {
        "chunk_size": 512,
        "chunk_overlap": 64,
        "top_k": 5,
        "retrieval_mode": "hybrid",
        "embed_model": "nomic-embed-text",
        "llm_model": "llama3.2"
    }
    
    test_metrics = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.82,
        "context_recall": 0.78,
        "context_precision": 0.80
    }
    
    print("Logging test experiment...")
    run_id = tracker.log_experiment(test_config, test_metrics, "test-run")
    print(f"Run ID: {run_id}")
    
    print("\nGetting experiment runs...")
    runs = tracker.get_experiment_runs()
    print(f"Found {len(runs)} runs")
