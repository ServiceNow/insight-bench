import insightbench.benchmarks as benchmarks


def create_jsons(datadir):
    import glob

    datasets = glob.glob(f"{datadir}/flag-*.ipynb")
    for d in datasets:
        dataset_dict = benchmarks.extract_notebook_info(d)
        print("success:", d)


create_jsons("data/notebooks")
