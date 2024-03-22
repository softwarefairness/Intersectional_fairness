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
                    choices = ['adult', 'default', 'mep1', 'mep2','german','bank'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['rf', 'svm', 'lr'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'adult': ['sex','race'],  'default':['sex','age'], 'mep1': ['sex','race'],'mep2': ['sex','race'],'german': ['sex','age'],'bank': ['married','age']}

val_name = "fairhome2_{}_{}_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()

sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]

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

    # split training data and test data
    dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)

    scaler.fit(dataset_orig_train)
    dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
    dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

    X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
    y_train = copy.deepcopy(dataset_orig_train['Probability'])
    X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
    y_test = copy.deepcopy(dataset_orig_test['Probability'])

    X_test2 =  X_test.copy()
    X_test3 = X_test.copy()
    X_test4 = X_test.copy()

    for rown in range(len(X_test)):
        X_test2.loc[rown,sa0] = 1 - X_test.loc[rown,sa0]

        X_test3.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]

        X_test4.loc[rown, sa0] = 1 - X_test.loc[rown, sa0]
        X_test4.loc[rown, sa1] = 1 - X_test.loc[rown, sa1]

        # if X_test.loc[rown,sa0] + X_test.loc[rown,sa1] == 1:
        #     continue
        # if X_test.loc[rown,sa0] == 1 and X_test.loc[rown,sa1] ==1:
        #     X_test.loc[rown, sa0] = replace_for_11[0]
        #     X_test.loc[rown, sa1] = replace_for_11[1]
        # elif X_test.loc[rown, sa0] == 0 and X_test.loc[rown, sa1] == 0:
        #     X_test.loc[rown, sa0] = replace_for_00[0]
        #     X_test.loc[rown, sa1] = replace_for_00[1]

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)

    pred_de1 = clf.predict_proba(X_test)
    pred_de2 = clf.predict_proba(X_test2)
    pred_de3 = clf.predict_proba(X_test3)
    pred_de4 = clf.predict_proba(X_test4)

    res = []
    for i in range(len(pred_de1)):
        prob_t = (pred_de1[i][1] + pred_de2[i][1] + pred_de3[i][1] +pred_de4[i][1])/4
        if prob_t >= 0.5:
            res.append(1)
        else:
            res.append(0)

    round_result = measure_final_score(dataset_orig_test, y_test, np.array(res), sa0, sa1)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()