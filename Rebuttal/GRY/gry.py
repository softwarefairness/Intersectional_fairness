import sys
import os
sys.path.append("../")
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility import get_classifier
from sklearn.model_selection import train_test_split
import argparse
from aif360.datasets import BinaryLabelDataset
from gerryfair import model as gmodel
import copy

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['adult', 'default','mep1', 'mep2','german'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'adult': ['sex','race'], 'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race'],'german': ['sex','age']}

val_name = "gry_{}_{}_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1}]
unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0}]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'wcspd-00','wcspd-01','wcspd-10','wcspd-11','wcspd', 'wcaod-00','wcaod-01','wcaod-10','wcaod-11','wcaod', 'wceod-00','wceod-01','wceod-10','wceod-11','wceod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20
for r in range(repeat_time):
    print (r)

    np.random.seed(r)
    #split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
    y_train = copy.deepcopy(dataset_orig_train['Probability'])
    X_prime_train = copy.deepcopy(dataset_orig_train.loc[:, macro_var[dataset_used]])

    X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
    y_test = copy.deepcopy(dataset_orig_test['Probability'])
    X_prime_test = copy.deepcopy(dataset_orig_test.loc[:, macro_var[dataset_used]])

    C = 15
    printflag = True
    gamma = .01
    fair_model = gmodel.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP')
    max_iters = 30
    fair_model.set_options(max_iters=max_iters)

    [errors, fp_difference] = fair_model.train(X_train, X_prime_train, y_train)

    predictions = fair_model.predict(X_test)
    predictions = [1 if value >= 0.5 else 0 for value in predictions]
    print(predictions)
#     dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
#                                             label_names=['Probability'],
#                                             protected_attribute_names=macro_var[dataset_used])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=macro_var[dataset_used])

    test_df_copy = dataset_orig_test.copy(deepcopy=True)
    test_df_copy.labels = np.array(predictions).reshape(-1, 1)

    round_result= measure_final_score(dataset_orig_test,test_df_copy,macro_var[dataset_used])
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()
