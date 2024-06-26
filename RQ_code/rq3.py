import sys
sys.path.append("../")
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from Fairea_multi.fairea import normalize,classify_region
from shapely.geometry import LineString

approach_list = ['rew','adv','eop','fairsmote', 'maat','fairmask', 'fairhome']
task_list = ['adult', 'compas_new', 'default', 'mep1', 'mep2','german']
model_list = ['lr','rf','svm', 'dl']
base_points = {}
for i in model_list:
    base_points[i]={}
    for j in task_list:
        base_points[i][j]={}
base_map = {1:'Acc', 2: 'Mac-P', 3: 'Mac-R', 4: 'Mac-F1', 5: 'MCC', 6: 'SPD', 7: 'AOD', 8:'EOD'}
for dataset in task_list:
    for i in model_list:
        fin = open('../Fairea_baseline_multi/'+dataset+'_'+i+'_baseline','r')
        count = 0
        for line in fin:
            count += 1
            if count in base_map:
                base_points[i][dataset][base_map[count]] = np.array(list(map(float,line.strip().split('\t')[1:])))
        fin.close()


data = {}
for i in model_list:
    data[i]={}
    for j in task_list:
        data[i][j]={}
        for k in ['Acc','Mac-P','Mac-R','Mac-F1','MCC','SPD','AOD','EOD']:
            data[i][j][k]={}

data_key_value_used = {1:'Acc', 2: 'Mac-R', 3: 'Mac-P', 4: 'Mac-F1', 5: 'MCC',28:'SPD', 33:'AOD',38:'EOD'}
for j in model_list:
    for name in ['default','rew','eop','fairsmote', 'maat','fairmask',  'fairhome']:
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

data_key_value_used = {1:'Acc', 2: 'Mac-R', 3: 'Mac-P', 4: 'Mac-F1', 5: 'MCC', 41:'SPD', 50:'AOD',59:'EOD'}
for j in model_list:
    for name in ['default','rew','eop','fairsmote', 'maat','fairmask',  'fairhome']:
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
                for j in model_list:
                    data[j][dataset][data_key_value_used[count]][name]=list(map(float,line.strip().split('\t')[1:]))
        fin.close()

region_count = {}
for dataset in task_list:
    region_count[dataset]={}
    for fairmetric in ['SPD','AOD','EOD']:
        region_count[dataset][fairmetric] = {}
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            region_count[dataset][fairmetric][permetric]={}
            for algo in model_list:
                region_count[dataset][fairmetric][permetric][algo]={}
                for name in approach_list:
                    region_count[dataset][fairmetric][permetric][algo][name]={}
                    for region_kind in ['good','win-win','bad','lose-lose','inverted']:
                        region_count[dataset][fairmetric][permetric][algo][name][region_kind]=0

for i in model_list:
    for j in task_list:
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


fout = open('rq3_result','w')
fout.write("-------Results for Figure2\n")
fout.write('\twin-win\tgood\tpoor\tinverted\tlose-lose\n')
for name in approach_list:
    fout.write(name)
    final_sum = 0
    final_count = {}
    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
        final_count[region_kind] = 0
    for fairmetric in ['SPD', 'AOD', 'EOD']:
        for permetric in ['Acc','Mac-P','Mac-R','Mac-F1','MCC']:
            for j in task_list:
                for i in ['rf', 'lr', 'svm','dl']:
                    for region_kind in ['good', 'win-win', 'bad', 'lose-lose', 'inverted']:
                        final_count[region_kind] += region_count[j][fairmetric][permetric][i][name][region_kind]
    for region_kind in ['lose-lose', 'bad', 'inverted', 'good', 'win-win']:
        final_sum += final_count[region_kind]
    for region_kind in ['win-win','good','bad','inverted','lose-lose']:
        fout.write('\t%f' % (final_count[region_kind]/final_sum))
    fout.write('\n')
fout.close()
