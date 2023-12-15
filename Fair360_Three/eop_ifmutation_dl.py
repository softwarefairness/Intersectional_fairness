import sys
import os
sys.path.append(os.path.abspath('.'))
from Measure_new import measure_final_score
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from utility_dl import get_classifier
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
import argparse
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing.eq_odds_postprocessing import EqOddsPostprocessing
import tensorflow as tf
import copy

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['compas_new'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['dl'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'compas_new': ['sex','race','age']}

val_name = "eopifmutation_{}_{}_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]
sa2 = macro_var[dataset_used][2]

privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1}, {macro_var[dataset_used][2]: 1}]
unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0}, {macro_var[dataset_used][2]: 0}]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'spd2-0','spd2-1', 'spd2', 'aod2-0','aod2-1', 'aod2', 'eod2-0','eod2-1','eod2','wcspd-000','wcspd-010','wcspd-100','wcspd-110','wcspd-001','wcspd-011','wcspd-101','wcspd-111','wcspd', 'wcaod-000','wcaod-010','wcaod-100','wcaod-110','wcaod-001','wcaod-011','wcaod-101','wcaod-111','wcaod', 'wceod-000','wceod-010','wceod-100','wceod-110','wceod-001','wceod-011','wceod-101','wceod-111','wceod']
for p_index in performance_index:
    results[p_index] = []

randseed = 12345679
repeat_time = 20
for r in range(repeat_time):
    print (r)
    np.random.seed(r)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    X_test = copy.deepcopy(dataset_orig_test)

    for rown in range(len(X_test)):
        if X_test.loc[rown, sa0] == 1 and X_test.loc[rown, sa1] == 0 and X_test.loc[rown, sa2] == 1:
            X_test.loc[rown, sa0] = 0
            X_test.loc[rown, sa1] = 0
            X_test.loc[rown, sa2] = 1
        elif X_test.loc[rown, sa0] == 0 and X_test.loc[rown, sa1] == 0 and X_test.loc[rown, sa2] == 0:
            X_test.loc[rown, sa0] = 1
            X_test.loc[rown, sa1] = 0
            X_test.loc[rown, sa2] = 0
        elif X_test.loc[rown, sa0] == 1 and X_test.loc[rown, sa1] == 1 and X_test.loc[rown, sa2] == 1:
            X_test.loc[rown, sa0] = 0
            X_test.loc[rown, sa1] = 0
            X_test.loc[rown, sa2] = 1
        elif X_test.loc[rown, sa0] == 0 and X_test.loc[rown, sa1] == 1 and X_test.loc[rown, sa2] == 1:
            X_test.loc[rown, sa0] = 0
            X_test.loc[rown, sa1] = 0
            X_test.loc[rown, sa2] = 1
        elif X_test.loc[rown, sa0] == 0 and X_test.loc[rown, sa1] == 1 and X_test.loc[rown, sa2] == 0:
            X_test.loc[rown, sa0] = 1
            X_test.loc[rown, sa1] = 0
            X_test.loc[rown, sa2] = 0
        elif X_test.loc[rown, sa0] == 1 and X_test.loc[rown, sa1] == 1 and X_test.loc[rown, sa2] == 0:
            X_test.loc[rown, sa0] = 1
            X_test.loc[rown, sa1] = 0
            X_test.loc[rown, sa2] = 0

    dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train,
                                            label_names=['Probability'],
                                            protected_attribute_names=macro_var[dataset_used])
    X_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=X_test,
                                label_names=['Probability'],
                                protected_attribute_names=macro_var[dataset_used])
    dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                           label_names=['Probability'],
                                           protected_attribute_names=macro_var[dataset_used])

    clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
    clf.fit(dataset_orig_train.features, dataset_orig_train.labels, epochs=20)

    train_pred = clf.predict_classes(dataset_orig_train.features).reshape(-1, 1)
    train_prob = clf.predict(dataset_orig_train.features).reshape(-1, 1)

    pred = clf.predict_classes(X_test.features).reshape(-1, 1)
    pred_prob = clf.predict(X_test.features).reshape(-1, 1)

    dataset_orig_train_pred = dataset_orig_train.copy()
    dataset_orig_train_pred.labels = train_pred
    dataset_orig_train_pred.scores = train_prob

    dataset_orig_test_pred = X_test.copy()
    dataset_orig_test_pred.labels = pred
    dataset_orig_test_pred.scores = pred_prob
    
    eqo = EqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     seed=randseed)
    eqo = eqo.fit(dataset_orig_train, dataset_orig_train_pred)
    pred_eqo = eqo.predict(dataset_orig_test_pred)

    data_tmp = dataset_orig_test.copy(deepcopy=True)
    data_tmp.labels = pred_eqo.labels

    round_result = measure_final_score(dataset_orig_test, data_tmp, macro_var[dataset_used])
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()