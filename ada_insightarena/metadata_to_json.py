import json

def merge_skill_into_question(
    question_file,   # e.g. "results_toy2_question.json"
    skill_file,      # e.g. "results_toy2_skill.json"
    output_file      # e.g. "results_toy2_merged.json"
):
    """
    Merges 'skill' from skill_file into the question_file structure:
    
    question_file has a structure like:
    {
      "dataset_66": {
        "method_1": {
          "exp_group": "insights_w_skills",
          "hash": "...",
          "questions": {
            "0": {
              "question": "...",
              "insight": "...",
              ...
            },
            "1": { ... }
          }
        },
        "method_2": {
          "exp_group": "insights_wo_skills",
          "hash": "...",
          "questions": {
            "0": { ... },
            ...
          }
        }
      },
      "dataset_67": {...},
      ...
    }

    skill_file has the same structure, but each question has "skill": "..."
    We copy that skill into the corresponding question in question_file.

    The final merged data is written to output_file.
    """

    # 1) Load both JSONs
    with open(question_file, "r", encoding="utf-8") as f_q:
        question_data = json.load(f_q)
    with open(skill_file, "r", encoding="utf-8") as f_s:
        skill_data = json.load(f_s)

    # 2) Merge skill from skill_data into question_data
    for dataset_key, dataset_val in question_data.items():
        # dataset_key looks like "dataset_66"
        # dataset_val is a dict with "method_1", "method_2", etc.
        if dataset_key not in skill_data:
            # If skill_data lacks this dataset, skip or handle gracefully
            continue

        skill_dataset_val = skill_data[dataset_key]

        # for each method ("method_1", "method_2", etc.)
        for method_key, method_val in dataset_val.items():
            if method_key not in skill_dataset_val:
                continue

            skill_method_val = skill_dataset_val[method_key]

            # both question_data[dataset_key][method_key] and skill_data[dataset_key][method_key]
            # should have a "questions" dict
            q_dict = method_val.get("questions", {})
            skill_q_dict = skill_method_val.get("questions", {})

            # Merge skill into each question index
            for q_index, q_content in q_dict.items():
                # skill_q_dict should have the same q_index
                if q_index in skill_q_dict:
                    skill_value = skill_q_dict[q_index].get("skill", None)
                    if skill_value is not None:
                        # Insert or overwrite the 'skill' field
                        q_content["skill"] = skill_value
                # else: no matching question index in skill file

    # 3) Write merged data to output_file
    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(question_data, out, indent=2, ensure_ascii=False)

    print(f"Merged skill from {skill_file} into {question_file}, wrote {output_file}")


# -------------------------------------------------------------------------
# Example usage if you run this script directly
# -------------------------------------------------------------------------
if __name__ == "__main__":
    merge_skill_into_question(
        question_file="results/results_pilot2ndrun_question.json",
        skill_file="results/results_pilot2ndrun_skill.json",
        output_file="results/results_pilot2ndrun_merged.json"
    )

# Load vis_id-based JSON (first JSON)
with open("results/results_pilot2ndrun_vis.json", "r") as f:
    vis_id_data = json.load(f)

# Load dataset_id-based JSON (second JSON)
with open("results/results_pilot2ndrun_merged.json", "r") as f:
    dataset_id_data = json.load(f)

# Merge metadata into dataset_id JSON
for vis_id, vis_info in vis_id_data.items():
    vis_num_str = vis_id.split("_")[1]  # "63"
    vis_num = int(vis_num_str)          # 63
    dataset_id = vis_num + 1            # 64
    dataset_id = f"dataset_{dataset_id}"
    # dataset_id = f"dataset_{vis_id.split('_')[1]}"  # Convert vis_10 -> dataset_10
    
    if dataset_id in dataset_id_data:  # Ensure matching dataset_id exists
        dataset_id_data[dataset_id]["metadata"] = vis_info["metadata"]  # Add metadata

# Save the merged JSON
with open("results/merged_pilot2ndrun_results.json", "w") as f:
    json.dump(dataset_id_data, f, indent=4)

print("Metadata merged successfully into dataset_id JSON!")
