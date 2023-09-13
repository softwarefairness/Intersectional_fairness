import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Fairea_multi.fairea import normalize,classify_region
from shapely.geometry import LineString

approach_list = ['rew', 'adv','eop','fairsmote', 'maat','fairmask', 'fairmutation', 'rewfairmutation', 'advfairmutation', 'eopfairmutation', 'fairsmotefairmutation', 'maatfairmutation']
base_points = {}
for i in ['rf','lr','svm', 'dl']:
    base_points[i]={}
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        base_points[i][j]={}

base_map = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 6: 'SPD', 7: 'AOD', 8:'EOD'}
for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
    for i in ['rf', 'lr', 'svm', 'dl']:
        fin = open('../Fairea_baseline_multi/'+dataset+'_'+i+'_baseline','r')
        count = 0
        for line in fin:
            count += 1
            if count in base_map:
                base_points[i][dataset][base_map[count]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()

data = {}
for i in ['rf','lr','svm','dl']:
    data[i]={}
    for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        data[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC','SPD','AOD','EOD']:
            data[i][j][k]={}

data_key_value_used = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 28: 'SPD', 33: 'AOD', 38:'EOD'}
for j in ['lr','rf','svm','dl']:
    for name in ['rew', 'eop','fairsmote', 'maat','fairmask', 'fairmutation', 'rewfairmutation', 'eopfairmutation', 'fairsmotefairmutation', 'maatfairmutation']:
        for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
            fin = open('../Results/'+name+'_'+j+'_'+dataset +'_multi.txt','r')
            count = 0
            for line in fin:
                count=count+1
                if count in data_key_value_used:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
            fin.close()
for name in ['adv','advfairmutation']:
    for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
        fin = open('../Results/'+name+'_lr_'+dataset +'_multi.txt','r')
        count = 0
        for line in fin:
            count=count+1
            if count in data_key_value_used:
                for j in ['lr', 'rf', 'svm','dl']:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

region_count = {}
for dataset in ['adult', 'compas', 'default', 'mep1', 'mep2']:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in ['rf','lr','svm','dl']:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in approach_list:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in ['rf','lr','svm','dl']:
    for j in ['adult', 'compas','default', 'mep1', 'mep2']:
        for fairmetric in ['SPD','AOD','EOD']:
            for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
                for name in approach_list:
                    methods = dict()
                    name_fair20 = data[i][j][fairmetric][name]
                    name_per20 = data[i][j][permetric][name]
                    for count in range(20):
                        methods[str(count)] = (float(name_per20[count]), float(name_fair20[count]))
                    normalized_accuracy, normalized_fairness, normalized_methods = normalize(base_points[i][j][permetric], base_points[i][j][fairmetric], methods)
                    baseline = LineString([(x, y) for x, y in zip(normalized_fairness, normalized_accuracy)])
                    mitigation_regions = classify_region(baseline, normalized_methods)
                    for count in mitigation_regions:
                        region_count[j][fairmetric][permetric][i][name][mitigation_regions[count]]+=1


fout = open('rq2_result','w')
fout.write("-------Results for Figure 3\n")
fout.write('\tProportion of cases surpassing the Fairea baseline\n')
for name in approach_list:
    fout.write(name)
    final_sum = 0
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for fairmetric in ['SPD', 'AOD', 'EOD']:
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            for j in ['adult', 'compas', 'default', 'mep1', 'mep2']:
                for i in ['rf', 'lr', 'svm','dl']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    fout.write('\t%f\n' % ((final_count['good']+final_count['win-win'])/final_sum))

fout.close()
