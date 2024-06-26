import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score, matthews_corrcoef
from aif360.metrics import ClassificationMetric
import copy

def cal_spd(dataset_test_pred, p_attr):
    labelname = 'Probability'
    # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    num1 = len(dataset_test_pred[(dataset_test_pred[p_attr] == 0)  & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[p_attr] == 0)])
    num2 = len(dataset_test_pred[(dataset_test_pred[p_attr] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[p_attr] == 1)])
    return [num1, num2, max([num1,num2])-min([num1,num2])]

def cal_eod(dataset_test, dataset_test_pred, p_attr):
    labelname = 'Probability'
    # dataset_test = dataset_test.convert_to_dataframe()[0]
    # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    num1 = len(dataset_test[(
            dataset_test[p_attr] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[
        labelname] == 1)])
    num2 = len(dataset_test[(
            dataset_test[p_attr] == 1)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[
            labelname] == 1)])

    return [num1, num2, max([num1,num2]) - min([num1,num2])]

def cal_aod(dataset_test, dataset_test_pred, p_attr):
    labelname = 'Probability'
    # dataset_test = dataset_test.convert_to_dataframe()[0]
    # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    num1 = len(dataset_test[(
            dataset_test[p_attr] == 0) & (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[p_attr] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[p_attr] == 0)  & (dataset_test[
            labelname] == 1)])

    num2 = len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 0)]) + len(dataset_test[(dataset_test[p_attr] == 1) & (
                                                                                                          dataset_test[labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(dataset_test[(dataset_test[p_attr] == 1) & (dataset_test[labelname] == 1)])

    return [num1, num2, (max([num1,num2]) - min([num1,num2]))/2]


def wc_spd(dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    attr3 = p_attrs[2]
    favorlabel = 1
    labelname = 'Probability'
    # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test_pred[labelname] = np.where(dataset_test_pred[labelname] == favorlabel, 1, 0)
    num1 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[attr3] == 0) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[attr3] == 0)])
    num2 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1)& (dataset_test_pred[attr3] == 0)])
    num3 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0)& (dataset_test_pred[attr3] == 0)])
    num4 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1)& (dataset_test_pred[attr3] == 0)])
    num5 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (
                dataset_test_pred[attr3] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[
            (dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[attr3] == 1)])
    num6 = len(dataset_test_pred[(dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1) & (
                dataset_test_pred[attr3] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[
            (dataset_test_pred[attr1] == 0) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[attr3] == 1)])
    num7 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0) & (
                dataset_test_pred[attr3] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[
            (dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 0) & (dataset_test_pred[attr3] == 1)])
    num8 = len(dataset_test_pred[(dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1) & (
                dataset_test_pred[attr3] == 1) & (dataset_test_pred[labelname] == 1)]) / len(
        dataset_test_pred[
            (dataset_test_pred[attr1] == 1) & (dataset_test_pred[attr2] == 1) & (dataset_test_pred[attr3] == 1)])
    return [num1, num2, num3, num4, num5, num6, num7, num8,  max([num1,num2,num3,num4,num5,num6,num7,num8])-min([num1,num2,num3,num4,num5,num6,num7,num8])]

def wc_aod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    attr3 =  p_attrs[2]
    favorlabel = 1
    labelname = 'Probability'
    # dataset_test = dataset_test.convert_to_dataframe()[0]
    # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred'+labelname] = np.where(dataset_test['pred'+labelname] == favorlabel, 1, 0)
    num_list = []
    num1 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0)& (dataset_test_pred[attr3] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0)& (dataset_test_pred[attr3] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num1)
    num2 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1)& (dataset_test_pred[attr3] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1)& (dataset_test_pred[attr3] == 0) & (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1)& (dataset_test_pred[attr3] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num2)
    num3 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0)& (dataset_test_pred[attr3] == 0) & (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0)& (dataset_test_pred[attr3] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num3)
    num4 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 0) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
            labelname] == 0)]) + len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
            labelname] == 1)])
    num_list.append(num4)
    num5 = len(dataset_test[(
                                    dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 0) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 0)]) + len(
        dataset_test[(
                             dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (
                                 dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                       labelname] == 1) & (
                                 dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num5)
    num6 = len(dataset_test[(
                                    dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 0) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 0)]) + len(
        dataset_test[(
                             dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (
                                 dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                       labelname] == 1) & (
                                 dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num6)
    num7 = len(dataset_test[(
                                    dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 0) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 0)]) + len(
        dataset_test[(
                             dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (
                                 dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                       labelname] == 1) & (
                                 dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num7)
    num8 = len(dataset_test[(
                                    dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 0) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 0)]) + len(
        dataset_test[(
                             dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (
                                 dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                       labelname] == 1) & (
                                 dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num8)
    return [num1, num2, num3, num4,num5,num6,num7,num8, (max(num_list) - min(num_list))/2]

def wc_eod(dataset_test, dataset_test_pred, p_attrs):
    attr1 = p_attrs[0]
    attr2 = p_attrs[1]
    attr3=p_attrs[2]
    favorlabel = 1
    labelname = 'Probability'
    # dataset_test = dataset_test.convert_to_dataframe()[0]
    # dataset_test_pred = dataset_test_pred.convert_to_dataframe()[0]
    dataset_test['pred'+labelname] = dataset_test_pred[labelname]
    dataset_test[labelname] = np.where(dataset_test[labelname] == favorlabel, 1, 0)
    dataset_test['pred' + labelname] = np.where(dataset_test['pred' + labelname] == favorlabel, 1, 0)
    num_list=[]
    num1 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred'+labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1)])
    num_list.append(num1)
    num2 = len(dataset_test[(
            dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
            labelname] == 1)])
    num_list.append(num2)
    num3 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 0)& (dataset_test_pred[attr3] == 0) & (dataset_test[
            labelname] == 1)])
    num_list.append(num3)
    num4 = len(dataset_test[(
            dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
        labelname] == 1) & (dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[(dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 0)& (dataset_test[
            labelname] == 1)])
    num_list.append(num4)
    num5 = len(dataset_test[(
                                    dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 1) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 0) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num5)
    num6 = len(dataset_test[(
                                    dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 1) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 0) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num6)
    num7 = len(dataset_test[(
                                    dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 1) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 1) & (dataset_test[attr2] == 0) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num7)
    num8 = len(dataset_test[(
                                    dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (
                                        dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                              labelname] == 1) & (
                                        dataset_test['pred' + labelname] == 1)]) / len(
        dataset_test[
            (dataset_test[attr1] == 1) & (dataset_test[attr2] == 1) & (dataset_test_pred[attr3] == 1) & (dataset_test[
                                                                                                             labelname] == 1)])
    num_list.append(num8)
    return [num1, num2, num3, num4,num5, num6,num7,num8, max(num_list) - min(num_list)]

def measure_final_score3(dataset_orig_test, y_test, y_pred, protected_attribute1, protected_attribute2,protected_attribute3):
    dataset_orig_predict = copy.deepcopy(dataset_orig_test)
    dataset_orig_predict['Probability'] = y_pred.reshape(-1,1)
    accuracy = accuracy_score(y_test, y_pred)
    recall_macro = recall_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score_macro = f1_score(y_test, y_pred, average='macro')
    mcc = matthews_corrcoef(y_test, y_pred)

    return [accuracy, recall_macro,  precision_macro,  f1score_macro, mcc]+ cal_spd(dataset_orig_predict, protected_attribute1) + cal_aod(dataset_orig_test, dataset_orig_predict, protected_attribute1) + cal_eod(dataset_orig_test, dataset_orig_predict, protected_attribute1) + cal_spd(dataset_orig_predict, protected_attribute2) + cal_aod(dataset_orig_test, dataset_orig_predict,protected_attribute2) + cal_eod(dataset_orig_test, dataset_orig_predict, protected_attribute2) + cal_spd(dataset_orig_predict, protected_attribute3) + cal_aod(dataset_orig_test, dataset_orig_predict,protected_attribute3) + cal_eod(dataset_orig_test, dataset_orig_predict, protected_attribute3)+ wc_spd(dataset_orig_predict, [protected_attribute1,protected_attribute2,protected_attribute3]) + wc_aod(dataset_orig_test, dataset_orig_predict, [protected_attribute1,protected_attribute2,protected_attribute3]) + wc_eod(dataset_orig_test, dataset_orig_predict, [protected_attribute1,protected_attribute2,protected_attribute3])