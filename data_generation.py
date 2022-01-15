# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from preprocess import parse_and_save_data
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from ast import literal_eval
from tqdm import tqdm
import pickle
import os
import requests
import zipfile

DIR_BASE = os.path.dirname(__file__)
DIR_GLOVE = os.path.join(DIR_BASE, "data", "glove")

GLOVE_PATH = os.path.join(DIR_GLOVE, "glove.840B.300d.txt")


def download_glove():
    if not os.path.exists(DIR_GLOVE):
        os.makedirs(DIR_GLOVE) 
        
    file_url = "http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip"
  
    r = requests.get(file_url, stream = True)
    total_length = int(r.headers.get('content-length'))

    zip_file_path = os.path.join(DIR_GLOVE, "glove.840B.300d.zip")
    
    with open(zip_file_path,"wb") as f, tqdm(
        desc=zip_file_path,
        total=total_length,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024):
             if chunk:
                 size = f.write(chunk)
                 bar.update(size)
                 
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(DIR_GLOVE)
        
    try:
        os.remove(zip_file_path)
    except: pass
    

def get_glove_embeddings():
    if not os.path.exists(GLOVE_PATH):
        print("Downlading Glove...")
        download_glove()
        
    glove_embeddings = {}
    
    with open(GLOVE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                glove_embeddings[word] = coefs
            except: pass
    
    print('Found %s word vectors.' % len(glove_embeddings))
    
    return glove_embeddings


def get_glove_embedding_matrix(emdim, embeddings_index, tokenizer):
    num_tokens = len(tokenizer.word_index) + 2
    
    hits = 0
    misses = 0
    
    embedding_matrix = np.zeros((num_tokens, emdim))
    for word, i in tqdm(tokenizer.word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector[:emdim]
            hits += 1
        else:
            misses += 1
    return embedding_matrix

def save_tokenizer(tokenizer):
    with open('data/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer():
    try:
        with open('data/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except: return None

def get_words_tokenizer(force_recreate = False):
    saved_tokenizer = load_tokenizer()
    
    if saved_tokenizer != None and not force_recreate:
        return saved_tokenizer
    
    tokenizer = Tokenizer()
    
    data_path = parse_and_save_data('train', 'train_data')

    data_frame = pd.read_csv(data_path,
                             converters={'context_tokens': literal_eval, 'question_tokens': literal_eval})

    docs = data_frame['context_tokens'].apply(lambda x: ' '.join(x)).to_list()
    docs.extend(data_frame['question_tokens'].apply(lambda x: ' '.join(x)).to_list())
    
    tokenizer.fit_on_texts(docs)
    
    save_tokenizer(tokenizer)
    
    return tokenizer

def get_tokenized_sequence(tokenizer, tokens_sequence, max_length):
    tokenized_sequence = tokenizer.texts_to_sequences([' '.join(line) for line in tokens_sequence])
    padded_sequence = pad_sequences(tokenized_sequence, maxlen=max_length, padding='post')
    return padded_sequence


def get_X(context_tokens, query_tokens, max_context_length, max_query_lenght):
    X = pd.DataFrame()
    
    tokenizer = get_words_tokenizer()

    tokenized_context = get_tokenized_sequence(tokenizer,
                                                 context_tokens, max_context_length)
    tokenized_question = get_tokenized_sequence(tokenizer,
                                               query_tokens, max_query_lenght)    
  
    X = np.array([[tokenized_context[i], tokenized_question[i]] for i in range(len(tokenized_context))])
    
    return X

def get_features(data_frame, max_context_length, max_query_lenght):
    X = get_X(data_frame['context_tokens'], data_frame['question_tokens'], max_context_length, max_query_lenght)
    
    data_frame = data_frame.where(pd.notnull(data_frame), None)
    
    answer_spans = data_frame['answer_token_idx_span'].apply(lambda x:  x if x != None else '(0, 0)').apply(
        lambda x: list(literal_eval(x))).to_list()
    
    y = np.array(answer_spans,
                                dtype='float32').clip(0, max_context_length - 1)
    
    return X, y


def input_format(context_tokens, query_tokens,max_context_length, max_query_lenght):
    X = get_X(context_tokens, query_tokens, max_context_length, max_query_lenght)
    
    return [np.stack(X[:,0], 0),
                    np.stack(X[:,1], 0)]
    
def get_train_data(max_context_length, max_query_lenght):
    data_path = parse_and_save_data('train', 'train_data')

    data_frame = pd.read_csv(data_path,
                             converters={'context_tokens': literal_eval, 'question_tokens': literal_eval})

    X, y = get_features(data_frame, max_context_length, max_query_lenght)
    
    train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.8, random_state = 5)
    
    train_x = [np.stack(train_x[:,0], 0),
                    np.stack(train_x[:,1], 0)]
    
    test_x = [np.stack(test_x[:,0], 0),
                    np.stack(test_x[:,1], 0)]
    
    return train_x, test_x, train_y, test_y


def get_validation_data(max_context_length, max_query_lenght):
    data_path = parse_and_save_data('dev', 'dev_data')
    
    data_frame = pd.read_csv(data_path,
                             converters={'context_tokens': literal_eval, 'question_tokens': literal_eval})
        
    return get_features(data_frame, max_context_length, max_query_lenght)
    