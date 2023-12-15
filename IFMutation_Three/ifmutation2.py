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
import copy

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
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


val_name = "ifmutation2_{}_{}_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()

sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]
sa2 = macro_var[dataset_used][2]

p111 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==1)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==1)])/len(dataset_orig[(dataset_orig[sa0]==1)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==1)])
p101 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==1)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==1)])/len(dataset_orig[(dataset_orig[sa0]==1)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==1)])
p011 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==0)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==1)])/len(dataset_orig[(dataset_orig[sa0]==0)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==1)])
p001 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==0)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==1)])/len(dataset_orig[(dataset_orig[sa0]==0)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==1)])
p110 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==1)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==0)])/len(dataset_orig[(dataset_orig[sa0]==1)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==0)])
p100 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==1)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==0)])/len(dataset_orig[(dataset_orig[sa0]==1)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==0)])
p010 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==0)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==0)])/len(dataset_orig[(dataset_orig[sa0]==0)&(dataset_orig[sa1]==1)&(dataset_orig[sa2]==0)])
p000 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==0)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==0)])/len(dataset_orig[(dataset_orig[sa0]==0)&(dataset_orig[sa1]==0)&(dataset_orig[sa2]==0)])

print(p111, p101,p011,p001, p110, p100, p010, p000)

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

    for rown in range(len(X_test)):
        if X_test.loc[rown,sa0] == 1 and X_test.loc[rown,sa1] ==0 and X_test.loc[rown,sa2] ==1:
            X_test.loc[rown, sa0] = 1
            X_test.loc[rown, sa1] = 1
            X_test.loc[rown, sa2] = 1
        elif X_test.loc[rown, sa0] == 0 and X_test.loc[rown, sa1] == 0 and X_test.loc[rown,sa2] ==0:
            X_test.loc[rown, sa0] = 0
            X_test.loc[rown, sa1] = 1
            X_test.loc[rown, sa2] = 0

    clf = get_classifier(clf_name)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    round_result = measure_final_score(dataset_orig_test, y_test, y_pred, sa0, sa1,sa2)
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()