# -*- coding: utf-8 -*-

from keras.layers import LSTM, Bidirectional, Embedding, Input, Softmax
from keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.initializers import Constant
from layers import SimilarityMatrixFormation, Megamerge,\
                    Probability, Output,\
                    ContextToQueryAttention, QueryToContextAttention,\
                    add_highway_layer
from loss_function import negative_avg_log_error
from data_generation import get_glove_embedding_matrix, get_glove_embeddings, get_words_tokenizer, input_format
from preprocess import tokenize
from accuracy_metric import accuracy
import tensorflow as tf
import matplotlib.pyplot as plt
import os

DIR_BASE = os.path.dirname(__file__)
DIR_CHECKPOINT = os.path.join(DIR_BASE, "checkpoint")
   
                
class BiDAFModel():
    
    def __init__(self, d, max_context_lenght, max_question_lenght, dropout = 0, learning_rate = 0.001, restore = True):
        """
        Parameters
        ----------
        d : int
            word embedding dimension.
        max_context_lenght : TYPE, optional
            passage lenght. The default is None.
        max_question_lenght : TYPE, optional
            query lenght. The default is None.
        dropout : float, optional
            the dropout for the encoder and the encoder layers. The default is 0.

        """
        
        self.d = d
        self.max_context_lenght = max_context_lenght
        self.max_question_lenght = max_question_lenght
        self.dropout = dropout
        
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        
        with strategy.scope():
            if restore:  
              self.make_or_restore_model()
            else:
               self.build_model()
            self.compile_model(learning_rate)
            
        
        self.model.summary()
        
    def make_or_restore_model(self):
        if not os.path.isdir(DIR_CHECKPOINT):
            os.mkdir(DIR_CHECKPOINT)

        checkpoints = [DIR_CHECKPOINT + "/" + name for name in os.listdir(DIR_CHECKPOINT)]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            print("Restoring from", latest_checkpoint)
            self.model = load_model(latest_checkpoint, custom_objects=self.get_custom_objects())
        else: 
            print("Creating a new model")
            self.build_model()

    def get_custom_objects(self):
        return {
            'SimilarityMatrixFormation': SimilarityMatrixFormation,
            'ContextToQueryAttention': ContextToQueryAttention,
            'QueryToContextAttention': QueryToContextAttention,
            'Megamerge': Megamerge,
            'Probability': Probability,
            'Output': Output,
            'negative_avg_log_error': negative_avg_log_error,
            'accuracy': accuracy
        }

        
    def build_model(self):
        passage_input_layer = Input(shape=(self.max_context_lenght, ), dtype='float32', name="passage_input_layer")
        question_input_layer = Input(shape=(self.max_question_lenght, ), dtype='float32', name="question_input_layer")

        tokenizer = get_words_tokenizer(True)
        vocab_size = len(tokenizer.word_index) + 2
        glove_embeddings = get_glove_embeddings()
        
        embedding_matrix = get_glove_embedding_matrix(self.d, glove_embeddings, tokenizer)
        
        passage_embedding_layer = Embedding(
            vocab_size,
            self.d,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=False)(passage_input_layer)
        
        question_embedding_layer = Embedding(
            vocab_size,
            self.d,
            embeddings_initializer=Constant(embedding_matrix),
            trainable=False)(question_input_layer)
        
        passage_highway_layer = add_highway_layer(passage_embedding_layer)
        question_highway_layer = add_highway_layer(question_embedding_layer)
        
        
        encoder = Bidirectional(
            LSTM(self.d, recurrent_dropout=self.dropout, return_sequences=True),
            name='bidirectional_encoder')
        
        passage_encoder_layer = encoder(passage_highway_layer)
        question_encoder_layer = encoder(question_highway_layer)
        
        similarity_matrix_formation_layer = SimilarityMatrixFormation(
            name='similarity_matrix_formation_layer')(
            [passage_encoder_layer, question_encoder_layer])

        
        context_to_query_attention_layer = ContextToQueryAttention(
            name='context_to_query_attention_layer')(
                [similarity_matrix_formation_layer, question_encoder_layer])
        
        query_to_context_attention_layer = QueryToContextAttention(
            name='query_to_context_attention_layer')(
                [similarity_matrix_formation_layer, passage_encoder_layer])
                
        mega_merge_layer = Megamerge(name='mega_merge_layer')(
            [passage_encoder_layer, context_to_query_attention_layer, query_to_context_attention_layer])
                
        
        M1 = Bidirectional(
            LSTM(self.d, recurrent_dropout=self.dropout, return_sequences=True),
            name='M1_layer')(mega_merge_layer)
        
        M2 = Bidirectional(
            LSTM(self.d, recurrent_dropout=self.dropout, return_sequences=True),
            name='M2_layer')(M1)
        
        
        p1 = Softmax(name='span_begin_probability_layer')(Probability()([mega_merge_layer, M1]))
        p2 = Softmax(name='span_end_probability_layer')(Probability()([mega_merge_layer, M2]))

        output = Output(name='output')([p1, p2])
        
        self.model = Model([passage_input_layer, question_input_layer], [output])
        
        
    def compile_model(self, learning_rate):
        self.model.compile(
            optimizer = Adam(learning_rate=learning_rate),
            loss = negative_avg_log_error,
            metrics = [accuracy],
        )
       
            
    def run_training(self, X, y, epochs=1, validation_data=None, batch_size=None):
        callbacks = [
            EarlyStopping(
                monitor="loss",
                mode="auto",
                patience=6,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath = os.path.join(DIR_CHECKPOINT, "ckpt-{epoch}"),
                save_freq = "epoch",
                save_best_only=True,
                monitor='val_accuracy',
                mode='max',
            )
        ]
        
        self.history = self.model.fit(
            X, y,
            validation_data = validation_data,
            epochs=epochs,
            callbacks=callbacks,
            verbose=True,
            use_multiprocessing=True,
            shuffle=True,
            batch_size=batch_size
        )

    def plot_history(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()    
            
        
    def evaluate(self, X, y, batch_size=128):
        self.model.evaluate(X, y, batch_size=batch_size)
        
        
    def predict(self, context, query, ids = None, batch_size=None):
        answers = []
        
        contexts_tokens = []
        querys_tokens = []
        
        if type(context) == list and type(query) == list and len(context) == len(query):
            for c in context:
                context_tokens = tokenize(c)
                contexts_tokens.append(context_tokens)
        
            for q in query:
                query_tokens = tokenize(q)
                querys_tokens.append(query_tokens)
                
        elif type(context) == str and type(query) == str:
            contexts_tokens.append(tokenize(context))
            querys_tokens.append(tokenize(query))
            context = [context, ]
            query = [query, ]

            
        else: raise ValueError("Invalid Input")
        
        X = input_format(contexts_tokens, querys_tokens,
                         self.max_context_lenght, self.max_question_lenght)
        
        y = self.model.predict(
            X,
            batch_size=batch_size,
            verbose=1,
            use_multiprocessing=True,
            )
            
        y_span_start = y[:, 0]
        y_span_end = y[:, 1]
        
        answers_span = []
        probabilities = []
        
        for i in range(len(contexts_tokens)):
            answer_span, probability = self.get_best_span(y_span_start[i, :], y_span_end[i, :],
                                                          len(contexts_tokens[i]))
            answers_span.append(answer_span)
            probabilities.append(probability)

        answers = []
        for i, answer_span in enumerate(answers_span):
            context_tokens = contexts_tokens[i]
            start, end = answer_span[0], answer_span[1]

            mapping = self.get_word_char_loc_mapping(
                context[i],
                context_tokens)

            char_loc_start = mapping[start]
            char_loc_end = mapping[end] + len(context_tokens[end])

            ans = context[i][char_loc_start:char_loc_end]

            results = {}
            
            if ids != None:
              results["id"] = ids[i]
            results["answer"] = ans
            results["span_begin"] = char_loc_start
            results["span_end"] = char_loc_end - 1
            results["probability"] = probabilities[i]

            answers.append(results)

        if type(context) == list:
            return answers
        else:
            return answers[0]
        
    def get_best_span(self, span_begin_probs, span_end_probs, context_length):
        max_span_probability = 0
        best_word_span = (0, 1)
    
        for i, p1 in enumerate(span_begin_probs):
            if i == 0:
                continue
    
            for j, p2 in enumerate(span_end_probs):
                if j > context_length - 1:
                    break
    
                if (j == 0) or (j < i):
                    continue
    
    
                if (p1 * p2) > max_span_probability:
                    best_word_span = (i, j)
                    max_span_probability = (p1 * p2)
    
            if span_begin_probs[0] * span_end_probs[0] > max_span_probability:
                best_word_span = (0, 0)
                max_span_probability = span_begin_probs[0] * span_end_probs[0]
    
        return best_word_span, max_span_probability
    
    
    
    def get_word_char_loc_mapping(self, context, context_tokens):
        mapping = {}
        idx = 0
        
        for i, word in enumerate(context_tokens):
            idx = context.find(word, idx)
    
            idx = idx
            mapping[i] = idx
    
    
        return mapping
        