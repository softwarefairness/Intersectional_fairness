import argparse
import numpy as np
import pandas as pd
import torch
import os

from model_configs import ModelConfigs
from models.infofair import InfoFair
from train_configs import TrainConfigs
from utils.custom_data_loader import CustomDataset  # 导入自定义的数据加载类
from utils.trainer import Trainer
from Measure_new import measure_final_score
from Measure_new_3 import measure_final_score3

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--enable_cuda", action="store_true", default=True, help="Enable CUDA training.")
parser.add_argument("--device_number", type=int, default=0, help="Which GPU to use for CUDA training.")
parser.add_argument("--datasets", type=str, nargs='+', default=["adult", "default", "mep1", "mep2", "german", "compas"], help="Datasets to train.")
parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs to train.")
parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
parser.add_argument("--regularization", type=float, default=0.1, help="Regularization hyperparameter.")
parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate (1 - keep probability).")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature annealing for gumbel softmax.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training.")
parser.add_argument("--repeat_time", type=int, default=20, help="Number of repetitions for training.")
parser.add_argument("--classifier", type=str, default="infofair", help="Classifier to use.")

args = parser.parse_args()
use_cuda = args.enable_cuda and torch.cuda.is_available()
device_name = f"cuda:{args.device_number}" if use_cuda else "cpu"

macro_var = {
    'adult': ['sex', 'race'],
    'default': ['sex', 'age'],
    'mep1': ['sex', 'race'],
    'mep2': ['sex', 'race'],
    'german': ['sex', 'age'],
    'compas': ['sex', 'race', 'age']
}

def train(model_config, train_config, dataset_used, repeat_time):
    if dataset_used == 'compas':
        results = {k: [] for k in [
            'accuracy', 'recall', 'precision', 'f1score', 'mcc',
            'spd0-0','spd0-1', 'spd0', 'aod0-0','aod0-1', 'aod0', 'eod0-0','eod0-1','eod0',
            'spd1-0','spd1-1', 'spd1', 'aod1-0','aod1-1', 'aod1', 'eod1-0','eod1-1','eod1',
            'spd2-0','spd2-1', 'spd2', 'aod2-0','aod2-1', 'aod2', 'eod2-0','eod2-1','eod2',
            'wcspd-000','wcspd-010','wcspd-100','wcspd-110','wcspd-001','wcspd-011','wcspd-101','wcspd-111','wcspd',
            'wcaod-000','wcaod-010','wcaod-100','wcaod-110','wcaod-001','wcaod-011','wcaod-101','wcaod-111','wcaod',
            'wceod-000','wceod-010','wceod-100','wceod-110','wceod-001','wceod-011','wceod-101','wceod-111','wceod'
        ]}
        sensitive_attrs = 'sex,race,age'
        measure_func = measure_final_score3
    else:
        results = {k: [] for k in [
            'accuracy', 'recall', 'precision', 'f1score', 'mcc',
            'spd0-0', 'spd0-1', 'spd0', 'aod0-0', 'aod0-1', 'aod0',
            'eod0-0', 'eod0-1', 'eod0', 'spd1-0', 'spd1-1', 'spd1',
            'aod1-0', 'aod1-1', 'aod1', 'eod1-0', 'eod1-1', 'eod1',
            'wcspd-00', 'wcspd-01', 'wcspd-10', 'wcspd-11', 'wcspd',
            'wcaod-00', 'wcaod-01', 'wcaod-10', 'wcaod-11', 'wcaod',
            'wceod-00', 'wceod-01', 'wceod-10', 'wceod-11', 'wceod'
        ]}
        sa0, sa1 = macro_var[dataset_used]
        sensitive_attrs = f"{sa0},{sa1}"
        measure_func = measure_final_score

    print("Dataset used: " + str(dataset_used))

    for r in range(repeat_time):
        print(f"Iteration: {r+1}/{repeat_time}")
        np.random.seed(r)

        # 使用自定义的数据集类，确保测试集不变
        data = CustomDataset(
            csv_file=f"./Dataset/{dataset_used}_processed.csv",
            sensitive_attrs=sensitive_attrs,
            device_name=device_name,
            random_seed=r
        )
        data.create_dataloader(train_batch_size=args.train_batch_size)

        # 初始化 InfoFair 模型
        model = InfoFair(
            nfeat=data.features.size()[1],
            nhids=model_config,
            nclass=data.num_classes,
            nsensitive=data.num_sensitive_groups,
            droprate=train_config["dropout"],
        )
        model.to(device_name)

        # 初始化 Trainer 并训练模型
        trainer = Trainer(
            train_config,
            model_config,
            data,
            model,
            torch.device(device_name),
        )
        trainer.train()
        trainer.test()

        pred = trainer.get_predictions()
        
        print(data.raw_test_data)
        if dataset_used == 'compas':
            round_result = measure_func(data.raw_test_data, data.test_data[1].cpu().numpy(), pred, 'sex', 'race', 'age')
        else:
            round_result = measure_func(data.raw_test_data, data.test_data[1].cpu().numpy(), pred, sa0, sa1)
            
        for i, k in enumerate(results.keys()):
            results[k].append(round_result[i])

    with open(f"./results/results_{dataset_used}.txt", 'w') as fout:
        for k, v in results.items():
            fout.write(f"{k}\t" + "\t".join(map(str, v)) + "\n")

if __name__ == "__main__":
    train_config = TrainConfigs().get_configs()
    train_config.update(vars(args))

    model_configs = ModelConfigs().get_configs()

    for dataset in args.datasets:
        # Update the dataset field in train_config dynamically
        train_config['dataset'] = dataset
        print(train_config["model"])
        train(
            model_config=model_configs[dataset],
            train_config=train_config,
            dataset_used=dataset,
            repeat_time=args.repeat_time,
        )
