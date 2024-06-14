import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

## Atomic fairness values were indeed calculated and stored in the Results and Results_Three file, so we just need to use the code to analzye
approach_list = ['fairhome']
task_list = ['adult', 'compas_new', 'default', 'german', 'mep1', 'mep2']
model_list = ['lr','rf','svm','dl']
data = {}
for i in model_list:
    data[i]={}
    for j in task_list:
        data[i][j]={}
        if j == 'compas_new':
            for k in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd2', 'aod2', 'eod2']:
                data[i][j][k]={}
        else:
            for k in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1']:
                data[i][j][k]={}


data_key_value_used = {8:'spd0', 11: 'aod0', 14: 'eod0', 17: 'spd1', 20:'aod1', 23: 'eod1'}
for j in model_list:
    for name in ['default', 'fairhome']:
        for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
            fin = open('../Results/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()

data_key_value_used = {8:'spd0', 11: 'aod0', 14: 'eod0', 17: 'spd1', 20:'aod1', 23: 'eod1', 26:'spd2', 29:'aod2', 33:'eod2'}
for j in model_list:
    for name in ['default','fairhome']:
        for dataset in ['compas_new']:
            fin = open('../Results_Three/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()

num_reduce = 0
num_incre = 0
num_equal = 0
for task in task_list:
    for j in model_list:
        for metric in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1']:
            if mann(data[j][task][metric]['default'], data[j][task][metric]['fairhome']) >= 0.05:
                num_equal+=1
            else:
                default_num = mean(data[j][task][metric]['default'])
                fh_num = mean(data[j][task][metric]['fairhome'])
                if default_num > fh_num:
                    num_incre += 1
                if default_num < fh_num:
                    num_reduce += 1
        if task == 'compas_new':
            for metric in ['spd2', 'aod2', 'eod2']:
                if mann(data[j][task][metric]['default'], data[j][task][metric]['fairhome']) >= 0.05:
                    num_equal+=1
                else:
                    default_num = mean(data[j][task][metric]['default'])
                    fh_num = mean(data[j][task][metric]['fairhome'])
                    if default_num > fh_num:
                        num_incre += 1
                    if default_num < fh_num:
                        num_reduce += 1

print("significantly improve fairness:", num_incre)
print("not significant:", num_equal)
print("significantly reduce fairness:", num_reduce)

