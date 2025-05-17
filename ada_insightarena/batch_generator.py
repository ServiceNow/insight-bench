import os
import shutil
import json

# Root folder containing:
#   - Numbered dataset folders (e.g., "1", "2", ...)
#   - Corresponding vis folders (e.g., "vis_0", "vis_1", ...)
root_datasets_dir = "results/insights_w_skills/Batch1"

# 1. Identify which indices are valid (i.e., both dataset and vis folders exist).
def filter_valid_indices(indices):
    valid = []
    for idx in indices:
        dataset_path = os.path.join(root_datasets_dir, str(idx))
        vis_path = os.path.join(root_datasets_dir, f"vis_{idx-1}")
        # Check that both dataset and vis folders exist and are directories
        if os.path.isdir(dataset_path) and os.path.isdir(vis_path):
            valid.append(idx)
    return valid

# 2. Create the pools, filtering out missing folders.
setA = filter_valid_indices(range(1, 50))    # 1..49
setB = filter_valid_indices(range(50, 66))   # 50..65
setC = filter_valid_indices(range(66, 101))  # 66..100

# 3. Determine how many full batches we can form with a 4–2–4 split.
num_batches = min(len(setA) // 4, len(setB) // 2, len(setC) // 4)

# 4. Contents for exp_dict.json (constant)
exp_dict_contents = {
    "challenge": "mid",
    "model": "gpt-4-0",
    "eval_mode": "insights",
    "with_skills": 0
}

# Helper function to create a batch folder, place exp_dict.json,
# and copy the dataset + vis folders for the given indices.
def create_batch(batch_folder_name, indices):
    # Create the batch folder
    os.makedirs(batch_folder_name, exist_ok=True)
    
    # Create exp_dict.json
    exp_dict_path = os.path.join(batch_folder_name, "exp_dict.json")
    with open(exp_dict_path, "w") as f:
        json.dump(exp_dict_contents, f, indent=4)
    
    # Copy dataset folders and corresponding vis folders
    for idx in indices:
        dataset_src = os.path.join(root_datasets_dir, str(idx))
        dataset_dst = os.path.join(batch_folder_name, str(idx))
        shutil.copytree(dataset_src, dataset_dst)
        
        vis_src = os.path.join(root_datasets_dir, f"vis_{idx-1}")
        vis_dst = os.path.join(batch_folder_name, f"vis_{idx-1}")
        shutil.copytree(vis_src, vis_dst)

# 5. Create full batches with 4–2–4.
for i in range(1, num_batches + 1):
    # Slice out the next 4 from setA, 2 from setB, 4 from setC
    batchA = setA[:4]
    setA   = setA[4:]
    
    batchB = setB[:2]
    setB   = setB[2:]
    
    batchC = setC[:4]
    setC   = setC[4:]
    
    batch_indices = batchA + batchB + batchC
    batch_folder = f"results/insights_w_skills_Batch{i}"
    
    create_batch(batch_folder, batch_indices)
    print(f"Created {batch_folder} with dataset indices: {batch_indices}")

# 6. Create one leftover batch if there are any indices left in setA, setB, or setC.
leftover_indices = setA + setB + setC
if leftover_indices:
    leftover_batch_folder = f"results/insights_w_skills_Batch{num_batches+1}_leftover"
    create_batch(leftover_batch_folder, leftover_indices)
    print(f"Created {leftover_batch_folder} with leftover dataset indices: {leftover_indices}")
