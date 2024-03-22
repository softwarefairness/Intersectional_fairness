import pandas as pd
import numpy as np
import copy,os
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from Measure_new import measure_final_score
import argparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

def get_classifier(name):
    if name == "lr":
        clf = LogisticRegression()
    elif name == "svm":
        clf = LinearSVC()
    elif name == "rf":
        clf = RandomForestClassifier()
    return clf

def reg2clf(protected_pred,threshold=.5):
    out = []
    for each in protected_pred:
        if each >=threshold:
            out.append(1)
        else: out.append(0)
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        choices=['compas_new'], help="Dataset name")
    parser.add_argument("-c", "--clf", type=str, required=True,
                        choices=['rf', 'svm', 'lr'], help="Classifier name")

    args = parser.parse_args()
    dataset_used = args.dataset
    clf_name = args.clf

    macro_var = {'compas_new': ['sex','race','age']}

    multi_attr = macro_var[dataset_used]

    val_name = "fairmask_{}_{}_multi.txt".format(clf_name, dataset_used)
    fout = open(val_name, 'w')

    results = {}
    performance_index =['accuracy', 'recall', 'precision', 'f1score', 'mcc', 'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0',  'eod0-0','eod0-1','eod0', 'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1', 'spd2-0','spd2-1', 'spd2', 'aod2-0','aod2-1', 'aod2', 'eod2-0','eod2-1','eod2','wcspd-000','wcspd-010','wcspd-100','wcspd-110','wcspd-001','wcspd-011','wcspd-101','wcspd-111','wcspd', 'wcaod-000','wcaod-010','wcaod-100','wcaod-110','wcaod-001','wcaod-011','wcaod-101','wcaod-111','wcaod', 'wceod-000','wceod-010','wceod-100','wceod-110','wceod-001','wceod-011','wceod-101','wceod-111','wceod']
    for p_index in performance_index:
        results[p_index] = []

    dataset_orig = pd.read_csv("../Dataset/" + dataset_used + "_processed.csv").dropna()

    repeat_time = 20
    for i in range(repeat_time):
        print(i)
        np.random.seed(i)

        dataset_orig_train, dataset_orig_test = train_test_split(dataset_orig, test_size=0.3, shuffle=True)
        scaler = MinMaxScaler()
        scaler.fit(dataset_orig_train)
        dataset_orig_train = pd.DataFrame(scaler.transform(dataset_orig_train), columns=dataset_orig.columns)
        dataset_orig_test = pd.DataFrame(scaler.transform(dataset_orig_test), columns=dataset_orig.columns)

        X_train = copy.deepcopy(dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'])
        y_train = copy.deepcopy(dataset_orig_train['Probability'])
        X_test = copy.deepcopy(dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'])
        y_test = copy.deepcopy(dataset_orig_test['Probability'])

        reduced = list(X_train.columns)
        reduced.remove(multi_attr[0])
        reduced.remove(multi_attr[1])
        reduced.remove(multi_attr[2])

        X_reduced, y_reduced0, y_reduced1, y_reduced2 = X_train.loc[:, reduced], X_train[multi_attr[0]],  X_train[multi_attr[1]], X_train[multi_attr[2]]
        # Build model to predict the protect attribute0
        clf1_0 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains0 = sm.fit_resample(X_reduced, y_reduced0)
        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf.fit(X_trains, y_trains0)
        y_proba = clf.predict_proba(X_trains)
        y_proba = [each[1] for each in y_proba]
        if isinstance(clf1_0, DecisionTreeClassifier) or isinstance(clf1_0, LogisticRegression):
            clf1_0.fit(X_trains, y_trains0)
        else:
            clf1_0.fit(X_trains, y_proba)

        # Build model to predict the protect attribute1
        clf1_1 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains1 = sm.fit_resample(X_reduced, y_reduced1)
        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf.fit(X_trains, y_trains1)
        y_proba = clf.predict_proba(X_trains)
        y_proba = [each[1] for each in y_proba]
        if isinstance(clf1_1, DecisionTreeClassifier) or isinstance(clf1_1, LogisticRegression):
            clf1_1.fit(X_trains, y_trains0)
        else:
            clf1_1.fit(X_trains, y_proba)

        # Build model to predict the protect attribute2
        clf1_2 = DecisionTreeRegressor()
        sm = SMOTE()
        X_trains, y_trains2 = sm.fit_resample(X_reduced, y_reduced2)
        clf = get_classifier(clf_name)
        if clf_name == 'svm':
            clf = CalibratedClassifierCV(base_estimator=clf)
        clf.fit(X_trains, y_trains2)
        y_proba = clf.predict_proba(X_trains)
        y_proba = [each[1] for each in y_proba]
        if isinstance(clf1_2, DecisionTreeClassifier) or isinstance(clf1_2, LogisticRegression):
            clf1_2.fit(X_trains, y_trains2)
        else:
            clf1_2.fit(X_trains, y_proba)


        X_test_reduced = X_test.loc[:, reduced]
        protected_pred0 = clf1_0.predict(X_test_reduced)
        protected_pred1 = clf1_1.predict(X_test_reduced)
        protected_pred2 = clf1_2.predict(X_test_reduced)
        if isinstance(clf1_0, DecisionTreeRegressor) or isinstance(clf1_0, LinearRegression):
            protected_pred0 = reg2clf(protected_pred0, threshold=0.5)
            protected_pred1 = reg2clf(protected_pred1, threshold=0.5)
            protected_pred2 = reg2clf(protected_pred2, threshold=0.5)

        # Build model to predict the target attribute Y
        clf2 = get_classifier(clf_name)
        clf2.fit(X_train, y_train)
        X_test.loc[:, multi_attr[0]] = protected_pred0
        X_test.loc[:, multi_attr[1]] = protected_pred1
        X_test.loc[:, multi_attr[2]] = protected_pred2
        y_pred = clf2.predict(X_test)

        round_result = measure_final_score(dataset_orig_test, y_test, y_pred, multi_attr[0], multi_attr[1], multi_attr[2])
        for i in range(len(performance_index)):
            results[performance_index[i]].append(round_result[i])

    for p_index in performance_index:
        fout.write(p_index)
        for i in range(repeat_time):
            fout.write('\t%f' % results[p_index][i])
        fout.write('\n')
    fout.close()
