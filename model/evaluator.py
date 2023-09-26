import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import labels

def evaluate(entities_list, entities_predicted_list):
    num_entities = 0
    num_predictions = 0


    l_correct = {v: 0 for k, v in labels.items() }
    l_total   = {v: 0 for k, v in labels.items() }


    for index, _ in enumerate(entities_list):
        for e in entities_list[index]:
            label = e['label']

            if label == 'country' or label == 'memo':
                continue

            find = False
            for pred in entities_predicted_list[index]:

                if \
                    pred['span'][0] == e['span'][0] and \
                    pred['span'][1] == e['span'][1] and \
                    pred['label'] == e['label']:

                    find = True
                    break

            if find:
                l_correct[label] = l_correct[label] + 1
            l_total[label] = l_total[label] + 1
    
    num_correct = sum(l_correct.values())


    for entities, entities_predicted in zip(entities_list, entities_predicted_list):

        ## TODO: 論文提出時、0 件で取れないものはスキップしている。
        entities = [e for e in entities if e['label'] != 'country' and e['label'] != 'memo']
        entities_predicted = [e for e in entities_predicted if e['label'] != 'country' and e['label'] != 'memo']

        # get_span_type = lambda e: (e['name'], e['span'][0], e['span'][1])
        # set_entities = set( get_span_type(e) for e in entities )
        # set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

        num_entities += len(entities)
        num_predictions += len(entities_predicted)

    precision = num_correct / num_predictions if num_predictions > 0 else 0
    recall = num_correct / num_entities if num_entities > 0 else 0
    f_value = 0
    if (precision + recall) > 0:
        f_value = 2 * precision * recall / (precision + recall)

    return {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value,
    }



# (name, span[0], span[1], label) が渡される前提
# label が None でない時は、そのラベルに対しておこなう
def evaluate_spacy(num_correct, entities_list, entities_predicted_list, label=None):
    num_entities = 0
    num_predictions = 0

    for entities, entities_predicted in zip(entities_list, entities_predicted_list):
        if  label:
            entities = [e for e in entities if e[3] == label]
            entities_predicted = [e for e in entities_predicted if e[3] == label]

        entities = [e for e in entities if e[2] != 'country' and e[2] != 'memo']
        entities_predicted = [e for e in entities_predicted if e[2] != 'country' and e[2] != 'memo']
        
        get_span_type = lambda e: (e[0], e[1], e[2])
        set_entities = set( get_span_type(e) for e in entities )
        set_entities_predicted = set( get_span_type(e) for e in entities_predicted )

        num_entities += len(entities)
        num_predictions += len(entities_predicted)

    precision = num_correct / num_predictions
    recall = num_correct / num_entities
    f_value = 0
    if (precision + recall) > 0:
        f_value = 2 * precision * recall / (precision + recall)

    return {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value,
    }
