# Parth Shah NLP A4 Inference

import sys
import re
from tensorflow import keras
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_SENT_LEN = 20
MAX_VOCAB_SIZE = 20000

# Reading the input word list.
def read_sens(sentences):
    inp = list()
    with open(sentences, "r") as f:
        all_lines = f.readlines()
    return [re.sub(r"[\.*]|[\,*]|[\'*]|[ *]|[\!*]|[\?*]|[\"*]|[#]|[\$]|[%]|[&]|[\(]|[\)]|[\*]|[\+]|[/]|[:]|[;]|[<]|[=]|[>]|[@]|[\[]|\\|[\]]|[\^]|[`]|[\{]|[\|]|[\}]|[~]|[-]|[\t]|[\n]", ' ', line).strip() for line in all_lines]

def seq_pad(data):
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(data) 
    data = tokenizer.texts_to_sequences(data)   
    data = pad_sequences(data, maxlen = MAX_SENT_LEN, padding='post', truncating='post')
    return data

# Classify based on model chosen
def classify(filtered_sens, model_type):
    model_dir = 'assignment4/data/nn_' + model_type + '.model'
    # load model 
    model = keras.models.load_model(model_dir)
    filtered_sens = seq_pad(filtered_sens)
    predictions = model.predict(filtered_sens)
    return predictions

if __name__ == '__main__':
    sentences = sys.argv[1]
    classifier_type = sys.argv[2]
    filtered_sens = read_sens(sentences)
    predictions = classify(filtered_sens, classifier_type)

    # print(predictions)
    for pred in predictions:
        # Use a 50% threshold
        if pred[1] > 0.5:
            print('positive')
        else:
            print('negative')
    
