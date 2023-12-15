import sys
import os
sys.path.append(os.path.abspath('.'))
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse
from Measure_new import measure_final_score
from aif360.datasets import BinaryLabelDataset
import copy

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True,
                    choices = ['compas_new'], help="Dataset name")
parser.add_argument("-c", "--clf", type=str, required=True,
                    choices = ['lr'], help="Classifier name")

args = parser.parse_args()

scaler = MinMaxScaler()
dataset_used = args.dataset
clf_name = args.clf

macro_var = {'compas_new': ['sex','race','age']}

val_name = "advifmutation_{}_{}_multi.txt".format(clf_name,dataset_used)
fout = open(val_name, 'w')

dataset_orig = pd.read_csv("../Dataset/"+dataset_used + "_processed.csv").dropna()
sa0 = macro_var[dataset_used][0]
sa1 = macro_var[dataset_used][1]
sa2 = macro_var[dataset_used][2]

# p11 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==1)&(dataset_orig[sa1]==1)])/len(dataset_orig[(dataset_orig[sa0]==1)&(dataset_orig[sa1]==1)])
# p10 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==1)&(dataset_orig[sa1]==0)])/len(dataset_orig[(dataset_orig[sa0]==1)&(dataset_orig[sa1]==0)])
# p01 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==0)&(dataset_orig[sa1]==1)])/len(dataset_orig[(dataset_orig[sa0]==0)&(dataset_orig[sa1]==1)])
# p00 = len(dataset_orig[(dataset_orig['Probability']==1)&(dataset_orig[sa0]==0)&(dataset_orig[sa1]==0)])/len(dataset_orig[(dataset_orig[sa0]==0)&(dataset_orig[sa1]==0)])
# replace_for_11 = [1,0]
# if p11-p10 > p11-p01:
#     replace_for_11 = [0,1]
# replace_for_00 = [1,0]
# if p00-p10 > p00-p01:
#     replace_for_00 = [0,1]
privileged_groups = [{macro_var[dataset_used][0]: 1}, {macro_var[dataset_used][1]: 1}, {macro_var[dataset_used][2]: 1}]
unprivileged_groups = [{macro_var[dataset_used][0]: 0}, {macro_var[dataset_used][1]: 0}, {macro_var[dataset_used][2]: 0}]

results = {}
performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'spd2-0','spd2-1', 'spd2', 'aod2-0','aod2-1', 'aod2', 'eod2-0','eod2-1','eod2','wcspd-000','wcspd-010','wcspd-100','wcspd-110','wcspd-001','wcspd-011','wcspd-101','wcspd-111','wcspd', 'wcaod-000','wcaod-010','wcaod-100','wcaod-110','wcaod-001','wcaod-011','wcaod-101','wcaod-111','wcaod', 'wceod-000','wceod-010','wceod-100','wceod-110','wceod-001','wceod-011','wceod-101','wceod-111','wceod']
for p_index in performance_index:
    results[p_index] = []

repeat_time = 20
for r in range(repeat_time):
    print(r)
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

    tf.reset_default_graph()
    sess = tf.Session()
    scope = "clf"+str(r)
    adversarial  = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name=scope,
                          debias=True,
                          sess=sess)
    adversarial = adversarial.fit(dataset_orig_train)
    pred_ad = adversarial.predict(X_test)

    dataset_orig_test_pred = dataset_orig_test.copy(deepcopy=True)
    dataset_orig_test_pred.labels = pred_ad.labels

    round_result = measure_final_score(dataset_orig_test, dataset_orig_test_pred, macro_var[dataset_used])
    for i in range(len(performance_index)):
        results[performance_index[i]].append(round_result[i])

for p_index in performance_index:
    fout.write(p_index)
    for i in range(repeat_time):
        fout.write('\t%f' % results[p_index][i])
    fout.write('\n')
fout.close()
