port = 5555

import os, sys
import base64
from flask import Flask, request, jsonify
import spacy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import labels
from config import token

best_model_file = 'k-00'

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_data', 'k_fold', 'SpaCy', best_model_file)

nlp = spacy.load(model_dir)
ner = nlp.get_pipe('ner')

app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# test page
@app.route('/')
def route():
    return 'Hitohaku API Server Running(Spacy)'

@app.route("/", methods=["POST"])
def post():
    # authentication
    # request_token = request.json['token']
    # if token != request_token:
    #     return ''

    text = request.json['text']

    entities_predicted = []
    doc = nlp(text)
    for entity in doc.ents:
        entities_predicted.append({
            'name': entity.text,
            'span': [entity.start_char, entity.end_char],
            'label': entity.label_
            })

    return entities_predicted

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=port, debug=True)
