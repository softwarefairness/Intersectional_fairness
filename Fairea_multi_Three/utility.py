from aif360.datasets import BinaryLabelDataset
import pandas as pd

def get_data(dataset_used):
    if dataset_used == "compas_new":
        mutation_strategy = {"1": [0, 1]}
        dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()
        dataset_orig = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig,
                                          label_names=['Probability'],
                                          protected_attribute_names=['sex', 'race','age'])
    return dataset_orig, mutation_strategy
