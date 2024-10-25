def get_benchmark(dataset_type, datadir):
    """load a set of tables and a set of golden insights to evaluate the agent's performance"""

    if dataset_type == "toy":
        # load the first 3 notebooks from datadir
        return
    elif dataset_type == "standard":
        return load_flag_dataset()
    elif dataset_type == "full":
        return load_flag_dataset()
    else:
        raise ValueError(f"Dataset type {dataset_type} not supported")
