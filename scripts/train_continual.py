import os
import sys
# Add project root to python path so 'src' can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import pandas as pd
import numpy as np
from src.continual_learning.trainer import ContinualLearner

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_experiment():
    dataset_config = load_config("config/dataset.yaml")
    model_config = load_config("config/model.yaml")
    cl_config = load_config("config/continual_learning.yaml")
    
    # Load manifest
    manifest_path = "data/manifest/full_manifest.csv"
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}. Please run preprocess.py first.")
        return
        
    df = pd.read_csv(manifest_path, dtype={"machine_id": str})
    
    # We will split data into tasks
    tasks = dataset_config["tasks"]
    
    # Dictionaries to store train and test splits per task
    task_data = {}
    
    for task in tasks:
        task_id = task["id"]
        machine_ids = task["machine_ids"]
        # Filter dataframe for this task
        task_df = df[df["machine_id"].isin(machine_ids)].copy()
        
        # Split train/test (simple 80/20 split based on dataset_config)
        # To be robust, stratify by label
        train_dfs = []
        test_dfs = []
        for label in [0, 1]:
            subset = task_df[task_df["label"] == label]
            # Shuffle
            subset = subset.sample(frac=1, random_state=dataset_config["random_seed"]).reset_index(drop=True)
            n_train = int(len(subset) * dataset_config["train_ratio"])
            
            train_dfs.append(subset.iloc[:n_train])
            test_dfs.append(subset.iloc[n_train:])
            
        train_df = pd.concat(train_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        task_data[task_id] = {
            "name": task["name"],
            "train": train_df,
            "test": test_df
        }
        
    methods = ["Naive_FT", "Joint", "RESONA_Replay", "RESONA_NoReplay"]
    
    for method in methods:
        print(f"\n\n{'='*50}\nSTARTING EXPERIMENT: {method}\n{'='*50}")
        
        # Initialize Learner
        config = {**model_config, **cl_config}
        config["method"] = method
        if method == "RESONA_NoReplay":
            config["method"] = "RESONA_Replay" # Use same logic but 0 memory budget
            config["memory_budget_per_class"] = 0
            
        learner = ContinualLearner(config)
        
        # Accuracy matrix: row=task_tested, col=task_learned
        acc_matrix = np.zeros((len(tasks), len(tasks)))
        
        # For joint training, keep accumulating training data
        cumulative_train_df = pd.DataFrame()
        
        for current_task_idx, task in enumerate(tasks):
            task_id = task["id"]
            t_data = task_data[task_id]
            
            print(f"\n--- STARTING {t_data['name']} ---")
            
            if method == "Joint":
                cumulative_train_df = pd.concat([cumulative_train_df, t_data["train"]], ignore_index=True)
                learner.train_on_task(t_data["name"], cumulative_train_df)
            else:
                learner.train_on_task(t_data["name"], t_data["train"])
            
            # Evaluate on all tasks seen so far (and current)
            print("--- Regression Evaluation ---")
            for eval_idx in range(current_task_idx + 1):
                eval_task_id = tasks[eval_idx]["id"]
                eval_data = task_data[eval_task_id]
                acc = learner.evaluate(eval_data["test"])
                acc_matrix[eval_idx, current_task_idx] = acc
                print(f"Accuracy on {eval_data['name']}: {acc:.2f}%")
                
        # Calculate Metrics
        final_accs = acc_matrix[:, -1]
        avg_acc = np.mean(final_accs)
        
        forgetting_scores = []
        for i in range(len(tasks) - 1): # Skip the last task as it hasn't been forgotten
            best_acc = np.max(acc_matrix[i, :-1])
            forgetting = best_acc - acc_matrix[i, -1]
            forgetting_scores.append(forgetting)
            
        avg_forgetting = np.mean(forgetting_scores) if forgetting_scores else 0.0
        
        print(f"\nMethod: {method}")
        print(f"Average Final Accuracy: {avg_acc:.2f}%")
        print(f"Average Forgetting: {avg_forgetting:.2f}%")
        
        # Save results
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, f"results_{method}.txt")
        with open(results_file, "w") as f:
            f.write(f"Method: {method}\n")
            f.write(f"Average Final Accuracy: {avg_acc:.2f}%\n")
            f.write(f"Average Forgetting: {avg_forgetting:.2f}%\n")
            f.write("Accuracy Matrix:\n")
            f.write(str(np.round(acc_matrix, 2)) + "\n")
        
if __name__ == "__main__":
    run_experiment()
