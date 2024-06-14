import sys
import os
sys.path.append(os.path.abspath('.'))
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from Measure_new import measure_final_score
import argparse
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import copy

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = CalibratedClassifierCV(base_estimator = LinearSVC())
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['compas_new'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'compas_new': ['sex','race','age']}

val_name = "fairhome_{}_{}atomic_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()

sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]
sa2 = macro_var[dataset_used][2]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'spd2-0','spd2-1', 'spd2', 'aod2-0','aod2-1', 'aod2', 'eod2-0','eod2-1','eod2','wcspd-000','wcspd-010','wcspd-100','wcspd-110','wcspd-001','wcspd-011','wcspd-101','wcspd-111','wcspd', 'wcaod-000','wcaod-010','wcaod-100','wcaod-110','wcaod-001','wcaod-011','wcaod-101','wcaod-111','wcaod', 'wceod-000','wceod-010','wceod-100','wceod-110','wceod-001','wceod-011','wceod-101','wceod-111','wceod']
for p_index in performance_index:
    results[p_index] = []


repeat_time = 20
for r in range(repeat_time):
    print (r)
    np.random.seed(r)

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
    y_train = copy.deepcopy(dataset_orig_train['Probability'])
    X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
    y_test = copy.deepcopy(dataset_orig_test['Probability'])

    X_test2 = X_test.copy()
    X_test3 = X_test.copy()
    X_test4 = X_test.copy()
    # X_test5 = X_test.copy()
    # X_test6 = X_test.copy()
    # X_test7 = X_test.copy()
    # X_test8 = X_test.copy()

    for rown in range(len(X_test)):
        X_test2.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]

        X_test3.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]

        X_test4.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]


        # X_test5.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        # X_test5.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]
        #
        # X_test6.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        # X_test6.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]
        #
        # X_test7.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]
        # X_test7.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]
        #
        # X_test8.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        # X_test8.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]
        # X_test8.loc[rown, sa2] = 1 - X_test.loc[rown, sa2]


    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)
    pred_de1 = clf.predict(X_test)
    pred_de2 = clf.predict(X_test2)
    pred_de3 = clf.predict(X_test3)
    pred_de4 = clf.predict(X_test4)
    # pred_de5 = clf.predict(X_test5)
    # pred_de6 = clf.predict(X_test6)
    # pred_de7 = clf.predict(X_test7)
    # pred_de8 = clf.predict(X_test8)

    res = []
    for i in range(len(pred_de1)):
        count1 = pred_de1[i] + pred_de2[i] + pred_de3[i] + pred_de4[i]
        count0 = 4 - count1
        if count1 >= count0:
            res.append(1)
        else:
            res.append(0)

    round_result = measure_final_score(dataset_orig_test, y_test, np.array(res), sa0, sa1,sa2)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()