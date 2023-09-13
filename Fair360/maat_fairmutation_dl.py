import sys
import os
sys.path.append(os.path.abspath('.'))
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import copy
from WAE import data_dis
from aif360.datasets import BinaryLabelDataset
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
from utility_dl import get_classifier
from Measure_new import measure_final_score
import tensorflow as tf

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['adult', 'default','german', 'compas', 'mep1', 'mep2'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['dl'], help="Classifier name")

    args = parser.parse_args()
    dataset_used = args.dataset
    clf_name = args.clf

    macro_var = {'adult': ['sex', 'race'], 'compas': ['sex', 'race'], 'default': ['sex', 'age'],'german': ['sex', 'age'],
                  'mep1': ['sex', 'race'], 'mep2': ['sex', 'race']}

    val_name = "maatfairmutation_{}_{}_multi.txt".format(clf_name,dataset_used)
    fout = open(val_name, 'w')

    dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()

    sa0 = macro_var[dataset_used][0]
    sa1 = macro_var[dataset_used][1]

    p11 = len(
        dataset_orig[(dataset_orig['Probability'] == 1) & (dataset_orig[sa0] == 1) & (dataset_orig[sa1] == 1)]) / len(
        dataset_orig[(dataset_orig[sa0] == 1) & (dataset_orig[sa1] == 1)])
    p10 = len(
        dataset_orig[(dataset_orig['Probability'] == 1) & (dataset_orig[sa0] == 1) & (dataset_orig[sa1] == 0)]) / len(
        dataset_orig[(dataset_orig[sa0] == 1) & (dataset_orig[sa1] == 0)])
    p01 = len(
        dataset_orig[(dataset_orig['Probability'] == 1) & (dataset_orig[sa0] == 0) & (dataset_orig[sa1] == 1)]) / len(
        dataset_orig[(dataset_orig[sa0] == 0) & (dataset_orig[sa1] == 1)])
    p00 = len(
        dataset_orig[(dataset_orig['Probability'] == 1) & (dataset_orig[sa0] == 0) & (dataset_orig[sa1] == 0)]) / len(
        dataset_orig[(dataset_orig[sa0] == 0) & (dataset_orig[sa1] == 0)])
    replace_for_11 = [1, 0]
    if p11 - p10 > p11 - p01:
        replace_for_11 = [0, 1]
    replace_for_00 = [1, 0]
    if p00 - p10 > p00 - p01:
        replace_for_00 = [0, 1]

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
        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        dataset_orig_train_new_for_attr1 = data_dis(pd.DataFrame(dataset_orig_train),macro_var[dataset_used][0],dataset_used)
        dataset_orig_train_new_for_attr2 = data_dis(pd.DataFrame(dataset_orig_train), macro_var[dataset_used][1], dataset_used)

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test_1 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
        for rown in range(len(dataset_orig_test_1)):
            if dataset_orig_test_1.loc[rown, sa0] + dataset_orig_test_1.loc[rown, sa1] == 1:
                continue
            if dataset_orig_test_1.loc[rown, sa0] == 1 and dataset_orig_test_1.loc[rown, sa1] == 1:
                dataset_orig_test_1.loc[rown, sa0] = replace_for_11[0]
                dataset_orig_test_1.loc[rown, sa1] = replace_for_11[1]
            elif dataset_orig_test_1.loc[rown, sa0] == 0 and dataset_orig_test_1.loc[rown, sa1] == 0:
                dataset_orig_test_1.loc[rown, sa0] = replace_for_00[0]
                dataset_orig_test_1.loc[rown, sa1] = replace_for_00[1]

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr1)
        dataset_orig_train_new_for_attr1 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr1), columns=dataset_orig.columns)
        dataset_orig_test_2 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
        for rown in range(len(dataset_orig_test_2)):
            if dataset_orig_test_2.loc[rown, sa0] + dataset_orig_test_2.loc[rown, sa1] == 1:
                continue
            if dataset_orig_test_2.loc[rown, sa0] == 1 and dataset_orig_test_2.loc[rown, sa1] == 1:
                dataset_orig_test_2.loc[rown, sa0] = replace_for_11[0]
                dataset_orig_test_2.loc[rown, sa1] = replace_for_11[1]
            elif dataset_orig_test_2.loc[rown, sa0] == 0 and dataset_orig_test_2.loc[rown, sa1] == 0:
                dataset_orig_test_2.loc[rown, sa0] = replace_for_00[0]
                dataset_orig_test_2.loc[rown, sa1] = replace_for_00[1]

        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train_new_for_attr2)
        dataset_orig_train_new_for_attr2 = pd.DataFrame(scaler.transform(dataset_orig_train_new_for_attr2),
                                                        columns=dataset_orig.columns)
        dataset_orig_test_3 = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)
        for rown in range(len(dataset_orig_test_3)):
            if dataset_orig_test_3.loc[rown, sa0] + dataset_orig_test_3.loc[rown, sa1] == 1:
                continue
            if dataset_orig_test_3.loc[rown, sa0] == 1 and dataset_orig_test_3.loc[rown, sa1] == 1:
                dataset_orig_test_3.loc[rown, sa0] = replace_for_11[0]
                dataset_orig_test_3.loc[rown, sa1] = replace_for_11[1]
            elif dataset_orig_test_3.loc[rown, sa0] == 0 and dataset_orig_test_3.loc[rown, sa1] == 0:
                dataset_orig_test_3.loc[rown, sa0] = replace_for_00[0]
                dataset_orig_test_3.loc[rown, sa1] = replace_for_00[1]

        dataset_orig_train = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train, label_names=['Probability'],
                                 protected_attribute_names=macro_var[dataset_used])
        dataset_orig_train_new_for_attr1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr1,
                                                label_names=['Probability'],
                                                protected_attribute_names=[macro_var[dataset_used][0]])
        dataset_orig_train_new_for_attr2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_train_new_for_attr2,
                                                    label_names=['Probability'],
                                                    protected_attribute_names=[macro_var[dataset_used][1]])
        dataset_orig_test = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test,
                                               label_names=['Probability'],
                                               protected_attribute_names=macro_var[dataset_used])
        dataset_orig_test_1 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_1,
                                                label_names=['Probability'],
                                                protected_attribute_names=macro_var[dataset_used])
        dataset_orig_test_2 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_2,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[macro_var[dataset_used][0]])
        dataset_orig_test_3 = BinaryLabelDataset(favorable_label=1, unfavorable_label=0, df=dataset_orig_test_3,
                                                 label_names=['Probability'],
                                                 protected_attribute_names=[macro_var[dataset_used][1]])

        clf = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
        clf.fit(dataset_orig_train.features, dataset_orig_train.labels)

        clf2 = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
        clf2.fit(dataset_orig_train_new_for_attr1.features, dataset_orig_train_new_for_attr1.labels)

        clf3 = get_classifier(clf_name, dataset_orig_train.features.shape[1:])
        clf3.fit(dataset_orig_train_new_for_attr2.features, dataset_orig_train_new_for_attr2.labels)

        test_df_copy = copy.deepcopy(dataset_orig_test)
        pred_de1 = clf.predict(dataset_orig_test_1.features)
        pred_de2 = clf2.predict(dataset_orig_test_2.features)
        pred_de3 = clf3.predict(dataset_orig_test_3.features)

        res = []
        for i in range(len(pred_de1)):
            prob_t = (pred_de1[i]+pred_de2[i]+pred_de3[i])/3
            if prob_t >= 0.5:
                res.append(1)
            else:
                res.append(0)

        test_df_copy.labels = np.array(res).reshape(-1,1)

        round_result= measure_final_score(dataset_orig_test,test_df_copy,macro_var[dataset_used])
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()
