import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

approach_list = ['rew','adv','eop','fairsmote', 'maat','fairmask', 'fairhome']
task_list = ['adult', 'compas_new', 'default', 'german', 'mep1', 'mep2']
model_list = ['lr','rf','svm','dl']
data = {}
for i in model_list:
    data[i]={}
    for j in task_list:
        data[i][j]={}
        for k in ['accuracy','precision','recall','f1score','mcc','spd','aod','eod']:
            data[i][j][k]={}

data_key_value_used = {1:'accuracy', 2: 'recall', 3: 'precision', 4: 'f1score', 5: 'mcc',28:'spd', 33:'aod',38:'eod'}
for j in model_list:
    for name in ['default', 'rew', 'eop', 'fairsmote', 'maat','fairmask','fairhome']:
        for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
            fin = open('../Results/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv']:
    for dataset in ['adult', 'default', 'mep1', 'mep2','german']:
        fin = open('../Results/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in model_list:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

data_key_value_used = {1:'accuracy', 2: 'recall', 3: 'precision', 4: 'f1score', 5: 'mcc', 41:'spd', 50:'aod',59:'eod'}
for j in model_list:
    for name in ['default','rew','eop','fairsmote', 'maat','fairmask','fairhome']:
        for dataset in ['compas_new']:
            fin = open('../Results_Three/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv']:
    for dataset in ['compas_new']:
        fin = open('../Results_Three/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm','dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

fout = open('rq1_result','w')

fout.write("-------Results for TableII\n")
for task in task_list:
    fout.write("\\hline\n\\multirow{2}*{"+task+"}")
    for name in ['default', 'fairhome']:
        fout.write('&'+name)
        for j in ['lr', 'rf']:
            for metric in ['spd', 'aod', 'eod','accuracy', 'precision', 'recall', 'f1score', 'mcc']:
                fout.write('&%.3f' % mean(data[j][task][metric][name]))
        fout.write('\\\\\n')

for task in task_list:
    fout.write("\\hline\n\\multirow{2}*{"+task+"}")
    for name in ['default', 'fairhome']:
        fout.write('&'+name)
        for j in ['svm', 'dl']:
            for metric in ['spd', 'aod', 'eod','accuracy', 'precision', 'recall', 'f1score', 'mcc']:
                fout.write('&%.3f' % mean(data[j][task][metric][name]))
        fout.write('\\\\\n')

fout.write("\n\n-------Results for TableIII\n")
fout.write('\tOriginal\tFairHOME\tabsc\trelac\n')
for i in ['spd','aod','eod','accuracy','precision','recall','f1score','mcc']:
    fout.write(i)
    olist = []
    flist = []
    for dataset in task_list:
        for j in model_list:
            olist.append(mean(data[j][dataset][i]['default']))
            flist.append(mean(data[j][dataset][i]['fairhome']))
    abschange = mean(flist)-mean(olist)
    relachange = 100*(mean(flist)-mean(olist))/mean(olist)
    fout.write('\t%.3f\t%.3f\t%.3f\t%.1f%%\n' % (mean(olist),mean(flist),abschange,relachange))

fout.close()
