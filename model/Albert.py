import os, sys
import random
import numpy as np
import unicodedata
import itertools
import copy
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AlbertForTokenClassification, AlbertTokenizerFast
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import labels
from model.evaluator import evaluate
from model.Bert import NER_tokenizer_BIO
import warnings

num_entity_type = len(labels)
num_labels = num_entity_type * 2 + 1
type_id_dict = {}
for key, label in labels.items():
  type_id_dict[label] = key

MODEL_NAME = 'ken11/albert-base-japanese-v1-with-japanese-tokenizer'

class Albert:

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = AlbertForTokenClassification_pl.load_from_checkpoint(model_path)
        else:
            self.model = AlbertForTokenClassification_pl(
                MODEL_NAME,
                num_labels=num_labels,
                lr=1e-5
            )

        self.tokenizer = NER_tokenizer_BIO.from_pretrained(
            MODEL_NAME,
            num_entity_type=num_entity_type
        )
        self.train_batch_size = 24
        self.max_length = 512

    def train(self, save_path, train_data, val_size=0.2, max_epoch=20, patience=3):


        warnings.filterwarnings('ignore')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.bert_tc.to(device)

        # データセットの分割
        train_dataset = self.convert_data(train_data)
        random.shuffle(train_dataset)
        n = len(train_dataset)
        n_train = int(n*(1.0-val_size))
        dataset_train = train_dataset[:n_train]
        dataset_val = train_dataset[n_train:]
        dataset_train_for_loader = self.create_dataset(
            dataset_train
        )
        dataset_val_for_loader = self.create_dataset(
            dataset_val
        )

        dataloader_train = DataLoader(
            dataset_train_for_loader, batch_size=self.train_batch_size, shuffle=True
        )
        dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)

        checkpoint = pl.callbacks.ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            save_weights_only=True,
            dirpath=os.path.join(save_path)
        )

        early_stop_callback = pl.callbacks.EarlyStopping(
            min_delta=0.00,
            patience=patience,
            verbose=False,
            monitor='val_loss',
            mode='min',
        )

        # print(dataset_val_for_loader[0])
        trainer = pl.Trainer(
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            # devices= 1 if torch.cuda.is_available() else 0,
            devices=1,
            max_epochs=max_epoch,
            callbacks=[checkpoint, early_stop_callback],
            num_sanity_val_steps=0
        )

        trainer.fit(self.model, dataloader_train, dataloader_val)
        best_model_path = checkpoint.best_model_path
        # 訓練後はベストなものをロードしておく
        self.model = AlbertForTokenClassification_pl.load_from_checkpoint(best_model_path)
        print(best_model_path)


    def test(self, test_data):
        entities_list, entities_predicted_list = [], []

        test_dataset = self.convert_data(test_data)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for index, t in enumerate(test_dataset):

            text = t['text']
            encoding, spans = self.tokenizer.encode_plus_untagged(
                text, return_tensors='pt', max_length=self.max_length
            )
            encoding = { k: v.to(device) for k, v in encoding.items() } 

            self.model.bert_tc.to(device)
            with torch.no_grad():
                output = self.model.bert_tc(**encoding)
                scores = output.logits
                scores = scores[0].cpu().numpy().tolist()
                
            # 分類スコアを固有表現に変換する
            entities_predicted = self.tokenizer.convert_bert_output_to_entities(
                text, scores, spans
            )

            entities = t['entities']
            for e in entities:
                e['label'] = labels[e['type_id']]
            entities_list.append(entities)
            
            for e in entities_predicted:
                e['label'] = labels[e['type_id']]
            entities_predicted_list.append( entities_predicted )

        return evaluate(entities_list=entities_list, entities_predicted_list=entities_predicted_list)


    def convert_data(self, train_data):
        dataset = copy.deepcopy(train_data)
        for sample in dataset:
            sample['text'] = unicodedata.normalize('NFKC', sample['text'])
            for e in sample["entities"]:
                e['type_id'] = type_id_dict[e['label']]
                del e['label']

        return dataset

    def create_dataset(self, dataset):
        """
        データセットをデータローダに入力できる形に整形。
        """
        dataset_for_loader = []
        for sample in dataset:
            text = sample['text']
            entities = sample['entities']
            encoding = self.tokenizer.encode_plus_tagged(
                text, entities, max_length=self.max_length
            )
            encoding = { k: torch.tensor(v) for k, v in encoding.items() }
            dataset_for_loader.append(encoding)
        return dataset_for_loader


class AlbertForTokenClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = AlbertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)