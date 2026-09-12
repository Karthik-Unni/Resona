import pandas as pd
import yaml

def check_leakage():
    print("Checking for data leakage...")
    manifest_path = "data/manifest/full_manifest.csv"
    try:
        df = pd.read_csv(manifest_path)
    except FileNotFoundError:
        print("Manifest not found. Ensure preprocessing is done.")
        return

    # Check 1: Ensure disjoint machine IDs for each task based on config
    with open("config/dataset.yaml", "r") as f:
        dataset_config = yaml.safe_load(f)
        
    tasks = dataset_config["tasks"]
    machine_sets = []
    
    for t in tasks:
        m_ids = set(t["machine_ids"])
        for prev in machine_sets:
            intersection = m_ids.intersection(prev)
            if intersection:
                print(f"[FAIL] Data Leakage Detected: Machine IDs {intersection} appear in multiple tasks.")
                return
        machine_sets.append(m_ids)
        
    print("[PASS] Machine IDs are properly disjoint across tasks.")
    
    # Check 2: OOD Machine ID is completely isolated
    ood_ids = set(dataset_config["ood_machine_ids"])
    for m_set in machine_sets:
        if ood_ids.intersection(m_set):
            print(f"[FAIL] OOD Machine ID {ood_ids} leaked into training tasks!")
            return
            
    print("[PASS] OOD Machine IDs are properly held out.")
    
    # Check 3: Random Seed
    if dataset_config.get("random_seed") != 42:
        print(f"[WARNING] Random seed is {dataset_config.get('random_seed')}, expected 42.")
    else:
        print("[PASS] Random seed = 42")
        
    print("All leakage checks passed!")

if __name__ == "__main__":
    check_leakage()
