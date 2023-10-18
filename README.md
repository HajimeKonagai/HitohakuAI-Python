# Training and Validation

## Required Python packages

* numpy
* en-ginza
* chardet
* pytorch-lightning
* transformers
* fugashi
* pytorch-lightning transformers
* PyTorch

(* PyTorch is installed based on information at https://pytorch.org/)

## Training Data

Created using Tool https://github.com/HajimeKonagai/HitohakuAI-Laravel

or

Download this.

* Training Data (318.7KB) (without endangered species)
https://data.hitohaku-ai.jp/annotation.zip
* Artificial data (28MB) (without endangered species)
https://data.hitohaku-ai.jp/artificial.json



## Model comparison, Train and Validation

Place the teacher data annotation.json in "data/annotation.json".
(If evaluation/training is performed using artificial data, place the teacher data artificial.json in "data/artificial.json")


```
python ./validation/k_fold_cross_validation.py
```

### Artificial data training and validation (SpaCy only)
```
python ./validation/artificial.py
python ./validation/artificial_only.py
```


# Launch Server 

## Required Python packages

* flask
* flask_cors
* spacy
* ja-ginza
* fugashi
* ipadic

The "model_data" folder contains the trained data for each training.

By changing best_model_file = 'k-00' in server/api.py,
You can change the trained data to be used by changing the best_model_file = 'k-00' in server/api.py.

```
python server/api.py

```
will launch the API server.
Please refer to the documentation of each vendor when using CGI on a rental server.

It is recommended to enable authentication when publishing.
server/api.py # uncomment authentication,

config.py and set your own token string in token.
When sending a request to the API, set the "token" parameter to the string set above.
"token" parameter.

