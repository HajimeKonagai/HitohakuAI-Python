import os, sys
import json
import random
import csv
import numpy as np
from itertools import chain
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.SpaCy import SpaCy
from model.Bert import Bert
from model.Albert import Albert

import time

time_start = time.time()

file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'annotation.json')
annotation_data = json.load(open(file_path, 'r', encoding='utf-8'))
file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'sampling.json')
sampling_data = json.load(open(file_path, 'r', encoding='utf-8'))

random.seed(0)
random.shuffle(annotation_data)


# k_count = 2
# annotation_data = annotation_data[:5]
max_epoch = 20

""" サンプリングのみの際はくんれん結果はおなじなので、先に訓練を終わらす """
# train_data にサンプリングを追加
#train_data = splited_data[0:idx] + splited_data[idx+1:]
#train_data = list(chain.from_iterable(train_data)) # flatten
train_data =  sampling_data
print(len(train_data))

# init model
model = SpaCy()

# train model with earli stopping
model.train(
    save_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'model_data',
            'sampling_only',
            'SpaCy',
            'k-00'),
    train_data = train_data,
    val_size=0.1,
    max_epoch=max_epoch)

def k_fold_cross_validation(k_count):
    splited_data = np.array_split(annotation_data, k_count)

    evaluate_scores = []

    for idx in range(0, k_count):

        test_data  = splited_data[idx]


        # evaluate model
        evaluate_score = model.test(test_data)

        evaluate_scores.append(evaluate_score)

    return evaluate_scores


# idx = 0
# model = SpaCy(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_data', 'SpaCy',  f"k-{(idx):02}"))

for k_count in [5, 10]:

    spacy_evaluate_scores = k_fold_cross_validation(
        k_count,
    )



    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', f"trining_evaluate-sampling_only-{k_count}.csv"), 'w', newline="") as f:
        writer = csv.writer(f)

        writer.writerow(['k', k_count])

        for k, scores in {
            'SpaCy': spacy_evaluate_scores,
        }.items():
            writer.writerow([k])
            writer.writerow(('k', 'num_entities', 'num_predictions', 'num_correct', 'precision', 'recall', 'f_value'))
            sums = {
                'k': 'sum',
                'num_entities': 0,
                'num_predictions': 0,
                'num_correct': 0,
                'precision': 0,
                'recall': 0,
                'f_value': 0
            }
            for idx, score in enumerate(scores):
                row = [idx]
                for k, val in score.items():
                    row.append(val)
                    sums[k] = val + sums[k]
                writer.writerow(row)
            
            means = ['mean']
            for v in sums.values():
                if type(v) is str:
                    continue
                means.append(v / k_count)

            writer.writerow(sums.values())
            writer.writerow(means)
            writer.writerow([])

print('-'*10 + ' elapsed sec ' + '-'*10)
print( time.time() - time_start )