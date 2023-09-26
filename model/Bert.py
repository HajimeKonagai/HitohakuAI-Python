import os, sys
import random
import numpy as np
import unicodedata
import itertools
import copy
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import BertForTokenClassification, BertJapaneseTokenizer
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import labels
from model.evaluator import evaluate

import warnings

MODEL_NAME = ''
num_entity_type = len(labels)
num_labels = num_entity_type * 2 + 1
type_id_dict = {}
for key, label in labels.items():
  type_id_dict[label] = key

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

class Bert:

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = BertForTokenClassification_pl.load_from_checkpoint(model_path)
        else:
            self.model = BertForTokenClassification_pl(
                MODEL_NAME,
                num_labels=num_labels,
                lr=1e-5
            )

        self.tokenizer = NER_tokenizer_BIO.from_pretrained(
            MODEL_NAME,
            num_entity_type=num_entity_type
        )
        self.train_batch_size = 16
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
        self.model = BertForTokenClassification_pl.load_from_checkpoint(best_model_path)
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




class BertForTokenClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = BertForTokenClassification.from_pretrained(
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


# 以下 Albert でも使用

class NER_tokenizer_BIO(BertJapaneseTokenizer):
# class NER_tokenizer_BIO(AlbertTokenizerFast):

    # 初期化時に固有表現のカテゴリーの数`num_entity_type`を
    # 受け入れるようにする。
    def __init__(self, *args, **kwargs):
        self.num_entity_type = kwargs.pop('num_entity_type')
        super().__init__(*args, **kwargs)

    def encode_plus_tagged(self, text, entities, max_length):
        """
        文章とそれに含まれる固有表現が与えられた時に、
        符号化とラベル列の作成を行う。
        """
        # 固有表現の前後でtextを分割し、それぞれのラベルをつけておく。
        splitted = [] # 分割後の文字列を追加していく
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text':text[position:start], 'label':0})
            splitted.append({'text':text[start:end], 'label':label})
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ]

        # 分割されたそれぞれの文章をトークン化し、ラベルをつける。
        tokens = [] # トークンを追加していく
        labels = [] # ラベルを追加していく
        for s in splitted:
            tokens_splitted = self.tokenize(s['text'])
            label = s['label']
            if label > 0: # 固有表現
                # まずトークン全てにI-タグを付与
                labels_splitted =  \
                    [ label + self.num_entity_type ] * len(tokens_splitted)
                # 先頭のトークンをB-タグにする
                labels_splitted[0] = label
            else: # それ以外
                labels_splitted =  [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        # ラベルに特殊トークンを追加
        labels = [0] + labels[:max_length-2] + [0]
        labels = labels + [0]*( max_length - len(labels) )
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章をトークン化し、それぞれのトークンの文章中の位置も特定しておく。
        IO法のトークナイザのencode_plus_untaggedと同じ
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割
        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        """
        Viterbiアルゴリズムで最適解を求める。
        """
        m = 2*num_entity_type + 1
        penalty_matrix = np.zeros([m, m])
        for i in range(m):
            for j in range(1+num_entity_type, m):
                if not ( (i == j) or (i+num_entity_type == j) ): 
                    penalty_matrix[i,j] = penalty
        
        path = [ [i] for i in range(m) ]
        scores_path = scores_bert[0] - penalty_matrix[0,:]
        scores_bert = scores_bert[1:]

        for scores in scores_bert:
            assert len(scores) == 2*num_entity_type + 1
            score_matrix = np.array(scores_path).reshape(-1,1) \
                + np.array(scores).reshape(1,-1) \
                - penalty_matrix
            scores_path = score_matrix.max(axis=0)
            argmax = score_matrix.argmax(axis=0)
            path_new = []
            for i, idx in enumerate(argmax):
                path_new.append( path[idx] + [i] )
            path = path_new

        labels_optimal = path[np.argmax(scores_path)]
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        """
        文章、分類スコア、各トークンの位置から固有表現を得る。
        分類スコアはサイズが（系列長、ラベル数）の2次元配列
        """
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        # 特殊トークンに対応する部分を取り除く
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        # Viterbiアルゴリズムでラベルの予測値を決める。
        labels = self.Viterbi(scores, num_entity_type)

        # 同じラベルが連続するトークンをまとめて、固有表現を抽出する。
        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0: # 固有表現であれば
                if 1 <= label <= num_entity_type:
                     # ラベルが`B-`ならば、新しいentityを追加
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    # ラベルが`I-`ならば、直近のentityを更新
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities