def load_dataset(dataset_type):
    """load a set of tables and a set of golden insights to evaluate the agent's performance"""

    if dataset_type == "toy":
        return load_toy_dataset()
    elif dataset_type == "standard":
        return load_flag_dataset()
    elif dataset_type == "full":
        return load_flag_dataset()
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
