import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta
from scipy.stats import spearmanr

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

approach_list = ['fairhome']
task_list = ['adult', 'compas_new', 'default', 'german', 'mep1', 'mep2']
model_list = ['lr','rf','svm','dl']
data = {}
for i in model_list:
    data[i]={}
    for j in task_list:
        data[i][j]={}
        if j == 'compas_new':
            for k in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd2', 'aod2', 'eod2', 'spd', 'aod', 'eod']:
                data[i][j][k]={}
        else:
            for k in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1', 'spd', 'aod', 'eod']:
                data[i][j][k]={}


data_key_value_used = {8:'spd0', 11: 'aod0', 14: 'eod0', 17: 'spd1', 20:'aod1', 23: 'eod1', 28:'spd', 33:'aod', 38:'eod'}
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

data_key_value_used = {8:'spd0', 11: 'aod0', 14: 'eod0', 17: 'spd1', 20:'aod1', 23: 'eod1', 26:'spd2', 29:'aod2', 33:'eod2', 41:'spd', 50:'aod', 59:'eod'}
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

single_list = []
inter_list = []
for task in task_list:
    for j in model_list:
        for metric in ['spd0', 'aod0', 'eod0', 'spd1', 'aod1', 'eod1']:
            single_list.append(mean(data[j][task][metric]['default'])-mean(data[j][task][metric]['fairhome']))
            inter_list.append(mean(data[j][task][metric[:-1]]['default'])-mean(data[j][task][metric[:-1]]['fairhome']))
        if task == 'compas_new':
            for metric in ['spd2', 'aod2', 'eod2']:
                single_list.append(mean(data[j][task][metric]['default']) - mean(data[j][task][metric]['fairhome']))
                inter_list.append(mean(data[j][task][metric[:-1]]['default']) - mean(data[j][task][metric[:-1]]['fairhome']))

print(len(single_list))
print(len(inter_list))
print(spearmanr(single_list,inter_list))
