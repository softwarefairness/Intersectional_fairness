import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
from numpy import mean
import scipy.stats as stats
from cliffs_delta import cliffs_delta

def mann(x, y):
    return stats.mannwhitneyu(x, y)[1]

approach_list = ['rew', 'adv','eop','fairsmote', 'maat','fairmask', 'ifmutation', 'rewifmutation', 'advifmutation', 'eopifmutation', 'fairsmoteifmutation', 'maatifmutation']
data = {}
for i in ['rf','lr','svm','dl']:
    data[i]={}
    for j in ['adult', 'compas', 'default','mep1', 'mep2']:
        data[i][j]={}
        for k in ['accuracy','precision','recall','f1score','mcc','spd1','aod1','eod1', 'spd2','aod2','eod2','spd','aod','eod']:
            data[i][j][k]={}

data_key_value_used = {1:'accuracy', 2: 'precision', 3: 'recall', 4: 'f1score', 5: 'mcc', 8: 'spd1', 11:'aod1', 14:'eod1', 17: 'spd2',20:'aod2',23:'eod2',28:'spd', 33:'aod',38:'eod'}
for j in ['lr','rf','svm','dl']:
    for name in ['default', 'rew', 'eop','fairsmote', 'maat','fairmask', 'ifmutation', 'rewifmutation', 'eopifmutation', 'fairsmoteifmutation', 'maatifmutation']:
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            fin = open('../Results/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','advifmutation']:
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        fin = open('../Results/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm','dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()


fout = open('rq1_result','w')
fout.write("-------Results for Table2\n")
fout.write('\tfairness_significant\tfairness_significant_large_effect\n')
for name in approach_list:
    fout.write(name)
    countt = {}
    countt['sig'] = 0
    countt['sig_large'] = 0
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        for j in ['lr', 'rf', 'svm','dl']:
            for i in ['spd','aod','eod']:
                if mean(data[j][dataset][i][name]) < mean(data[j][dataset][i]['default']) and mann(data[j][dataset][i][name], data[j][dataset][i]['default']) < 0.05:
                    countt['sig']+=1
                    if abs(cliffs_delta(data[j][dataset][i][name], data[j][dataset][i]['default'])[0]) >=0.428:
                        countt['sig_large']+=1
    fout.write('\t%f\t%f\n' % (countt['sig']/60,countt['sig_large']/60))

fout.write("\n\n-------Results for Table3\n")
for i in ['spd', 'aod', 'eod']:
    fout.write(i+'\n')
    fout.write('\tWin\tTie\tLose\n')
    for name in ['rew', 'adv', 'eop', 'fairsmote', 'maat', 'fairmask']:
        count={}
        count['win'] = 0
        count['tie'] =0
        count['lose']=0
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            for j in ['lr', 'rf', 'svm', 'dl']:
               if mann(data[j][dataset][i][name], data[j][dataset][i]['ifmutation']) >= 0.05:
                   count['tie']+=1
               elif mean(data[j][dataset][i][name]) > mean(data[j][dataset][i]['ifmutation']):
                   count['win']+=1
               else:
                   count['lose']+=1
        fout.write(name+'\t'+str(count['win'])+'\t'+str(count['tie'])+'\t'+str(count['lose'])+'\n')


fout.write("\n\n-------Results for Table4\n")
for i in ['spd', 'aod', 'eod']:
    fout.write(i + '\n')
    fout.write('\tWin\tTie\tLose\n')
    for name in ['rewifmutation','advifmutation','eopifmutation','fairsmoteifmutation','maatifmutation']:
        count={}
        count['win'] = 0
        count['tie'] =0
        count['lose']=0
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            for j in ['lr', 'rf', 'svm', 'dl']:
               if mann(data[j][dataset][i]['ifmutation'], data[j][dataset][i][name]) >= 0.05:
                   count['tie']+=1
               elif mean(data[j][dataset][i]['ifmutation']) < mean(data[j][dataset][i][name]):
                   count['win']+=1
               else:
                   count['lose']+=1
        fout.write(name+'\t'+str(count['win'])+'\t'+str(count['tie'])+'\t'+str(count['lose'])+'\n')
fout.close()
