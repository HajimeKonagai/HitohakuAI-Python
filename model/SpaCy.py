import os, sys
import random
import numpy as np
import spacy
from spacy.training import Example
from spacy.util import minibatch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import labels
from model.evaluator import evaluate

import warnings

class SpaCy:

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = spacy.load(model_path)
        else:
            self.model = spacy.blank('ja')
            if 'ner' not in self.model.pipe_names:
                self.model.create_pipe('ner')
                ruler = self.model.add_pipe("ner", last=True)
                for index, label in labels.items():
                    ruler.add_label(label)

    def train(self, save_path, train_data, val_size=0.2, max_epoch=20, patience=3):

        warnings.filterwarnings('ignore')

        train_data = self.convert_data(train_data)

        early_stopping = EarlyStopping(patience=patience)

        other_pipes = [pipe for pipe in self.model.pipe_names if pipe != 'ner']
        with self.model.disable_pipes(*other_pipes): # only train ner
            optimizer = self.model.begin_training()

            examples = []
            for text, annots in train_data:
                examples.append(Example.from_dict(self.model.make_doc(text), annots))
            # self.model.initialize(lambda: examples)
            for i in range(max_epoch):

                random.shuffle(examples)
                losses = {}
                val_losses = {}

                N = int(np.floor(len(examples) * (1-val_size)))
                train = examples[:N]
                val   = examples[N:]

                for batch in minibatch(train, size=8):
                    self.model.update(batch, drop=0.0, sgd=optimizer, losses=losses)

                # validation loss (overhead)
                self.model.update(val, drop=0.0, sgd=None, losses=val_losses)

                if early_stopping(val_losses['ner']):
                    break
                else:
                    print(f"iteration: {i}, val_loss: {val_losses['ner']:.8f}")
                    # print(losses)
                    if not os.path.isdir(save_path):
                        os.makedirs(save_path)
                    self.model.to_disk(save_path)

    def test(self, test_data):
        entities_list, entities_predicted_list = [], []

        for index, t in enumerate(test_data):

            entities = t['entities']
            entities_list.append(entities)

            text = t['text']
            entities_predicted = []
            doc = self.model(text)
            for entity in doc.ents:
                if entity.label_ == 'memo' or entity.label_ == 'country':
                    continue
                entities_predicted.append({
                    'span': (entity.start_char, entity.end_char),
                    'label': entity.label_
                })

            entities_predicted_list.append(entities_predicted)

            # text_list.append(text)

        return evaluate(entities_list=entities_list, entities_predicted_list=entities_predicted_list)


    def convert_data(self, train_data):
        # SpaCy 方式に変換(位置だけで良い)
        dataset = []
        for data in train_data:
            entities = []
            for entity in data['entities']:
                entities.append((entity['span'][0], entity['span'][1], entity['label']))
            dataset.append((data['text'], {'entities': entities}))

        return dataset


class EarlyStopping:
    def __init__(self, patience=10):
        self.epoch = 0
        self.pre_loss = float('inf')
        self.patience = patience
        
    def __call__(self, current_loss):
        if self.pre_loss < current_loss:
            self.epoch += 1
            if self.epoch > self.patience:
                return True
        else:
            self.epoch = 0
            self.pre_loss = current_loss

        return False