# -*- coding: utf-8 -*-

import keras.backend as K
from keras.layers import Dense, Multiply, Add, Lambda, TimeDistributed
from keras.initializers import Constant
from keras.engine.base_layer import Layer
from keras.layers.advanced_activations import Softmax


def add_highway_layer(x, activation = 'relu', transform_gate_bias=-1):
    """ add keras layers to the model to represent Highway Layer.
    
    Args:
        activation: str ('relu' or 'tanh').
            the activation function for the affine transformation layer.
            
        transform_gate_bias: int
            inisialize value for the bias of transform gate layer
            
    """
    
    print(x)
    output_dim = K.int_shape(x)[-1]
    
    transform_gate_bias_initializer = Constant(transform_gate_bias)

    transform_gate_layer = TimeDistributed(Dense(units=output_dim,
                                 bias_initializer=transform_gate_bias_initializer,
                                 activation='sigmoid'))(x)
    
    affine_transformation_layer = TimeDistributed(Dense(units=output_dim,
                                activation=activation))(x)
    
    transformed_gated = Multiply()([transform_gate_layer,
                                        affine_transformation_layer])
    
    carry_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(output_dim,)
            )(transform_gate_layer)

    identity_gated = Multiply()([carry_gate, x])
    
    x = Add()([transformed_gated, identity_gated])
    
    return x



class SimilarityMatrixFormation(Layer):
    """ calculating the similarity matrix.
        input:
            H: (T, 2d) contextual embedding of the context.
            U: (J, 2d) contextual embedding of the query.
            
        output:
            S: (T, J).
    """
    
    def __init__(self, **kwargs):
        super(SimilarityMatrixFormation, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[0][-1]
        weight_vector_dim = d * 3
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(weight_vector_dim, 1),
                                      initializer='glorot_uniform',
                                      trainable=True)
        
        self.bias = self.add_weight(name='bias',
                                    shape=(),
                                    initializer='zeros',
                                    trainable=True)
        
        super(SimilarityMatrixFormation, self).build(input_shape)


    def get_concatenated_matrix(self, context_expansion, query_expansion):
        vectors_mul = context_expansion * query_expansion
        return K.concatenate([context_expansion, query_expansion, vectors_mul], -1)
        
    def calculate_similarity(self, context_expansion, query_expansion):
        concatenated_matrix = self.get_concatenated_matrix(context_expansion, query_expansion)
        dot = K.dot(concatenated_matrix, self.kernel)
        out =  K.squeeze(dot, axis=-1) + self.bias
        return out 

    def call(self, inputs):
        context_tokens, query_tokens = inputs
        
        T = K.shape(context_tokens)[1]
        J = K.shape(query_tokens)[1]
        
        context_expansion_dim = K.concatenate([[1, 1], [J], [1]], 0)
        query_expansion_dim = K.concatenate([[1], [T], [1, 1]], 0)
        
        context_expansion = K.tile(
            K.expand_dims(context_tokens, axis=2),
            context_expansion_dim)
        
        query_expansion = K.tile(
            K.expand_dims(query_tokens, axis=1),
            query_expansion_dim)
        
        return self.calculate_similarity(context_expansion, query_expansion)


    def compute_output_shape(self, input_shape):
        b = input_shape[0][0]
        T = input_shape[0][1]
        J = input_shape[1][1]
        return (b, T, J)
    
    
class ContextToQueryAttention(Layer):
    """ 
        input:
            S: (T, J) similarity matrix.
            U: (J, 2d) contextual embedding of the query.
            
        output:
            U^: (T, 2d).
    """
    
    def __init__(self, **kwargs):
        super(ContextToQueryAttention, self).__init__(**kwargs)

    def call(self, inputs):
        similarity_matrix, query_tokens = inputs
        
        attention_distribution = Softmax()(similarity_matrix)
                
        return K.sum(
            K.expand_dims(attention_distribution, axis=-1) * K.expand_dims(query_tokens, axis=1),
            -2)

    def compute_output_shape(self, input_shape):
        b = input_shape[0][0]
        T = input_shape[0][1]
        d = input_shape[1][2]
        
        return (b, T, d)



class QueryToContextAttention(Layer):
    """ 
        input:
            S: (T, J) similarity matrix.
            H: (T, 2d) contextual embedding of the context.
            
        output:
            H^: (T, 2d).
    """
    
    def __init__(self, **kwargs):
        super(QueryToContextAttention, self).__init__(**kwargs)

    def call(self, inputs):
        similarity_matrix, context_tokens = inputs
        
        max_similarity = K.max(similarity_matrix, axis=-1)
        
        attention_distribution = Softmax()(max_similarity)
        
        weighted_sum = K.sum(
            K.expand_dims(attention_distribution, axis=-1) * context_tokens,
            -2)
        
        T = K.shape(context_tokens)[1]
        
        dublicated_sum = K.tile(
            K.expand_dims(weighted_sum, 1),
            [1, T, 1])
        
        return dublicated_sum

    def compute_output_shape(self, input_shape):
        b = input_shape[0][0]
        T = input_shape[0][1]
        d = input_shape[1][2]
        
        return (b, T, d)

    
    
class Megamerge(Layer):
    """ merging the attention outputs in a single matrix.
    
        input:
            H: (T, 2d) contrxtual embedding of the context.
            U^: (T, 2d) context to query attention.
            H^: (T, 2d) query to context attention.
            
        output:
            G: (T, 8d) merged attention.
    """
    
    def __init__(self, **kwargs):
        super(Megamerge, self).__init__(**kwargs)

    def call(self, inputs):
        context_tokens, c2q_attention, q2c_attention = inputs
        
        context_c2q_mul = context_tokens * c2q_attention
        context_q2c_mul = context_tokens * q2c_attention
        
        G = K.concatenate(
            [context_tokens, c2q_attention, context_c2q_mul, context_q2c_mul],
            axis=-1)
        return G

    def compute_output_shape(self, input_shape):
        b = input_shape[0][0]
        T = input_shape[0][1]
        d = input_shape[0][2]
        
        return (b, T, d * 4)


class Probability(Layer):
    """ return a probability for every word 
        in the context being the first or last word in the answer.
        
        input:
            G: (T, 8d) merged attention.
            M: (T, 2d) the output of the decoder layer.
            
        output:
            p: (T, )
        
    """
    
    def __init__(self, **kwargs):
        super(Probability, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_vector_dim = input_shape[0][-1] + input_shape[1][-1]
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(weight_vector_dim, 1),
                                      initializer='glorot_uniform',
                                      trainable=True)
        
        self.bias = self.add_weight(name='bias',
                                    shape=(),
                                    initializer='zeros',
                                    trainable=True)
        
        super(Probability, self).build(input_shape)

    def calculate_probabilities(self, concatenated_matrix):
        dot = K.dot(concatenated_matrix, self.kernel)
        out = K.squeeze(dot, axis=-1) + self.bias
        return out
        
    def call(self, inputs):
        G, M = inputs
        
        concatenation = K.concatenate([G, M])
        return self.calculate_probabilities(concatenation)


    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1]
    
    

class Output(Layer):
    """ conmbining the outputs in one matrix.
    
        input:
            p1: (T, ) the probability that each word is the first word in the answer span.
            p2: (T, ) the probability that each word is the last word in the answer span.
            
        output:
            (2, T).
    """
    
    def __init__(self, **kwargs):
        super(Output, self).__init__(**kwargs)

    def call(self, inputs):
        p1, p2 = inputs
        return K.stack([p1, p2], axis = 1)

    def compute_output_shape(self, input_shape):
        b = input_shape[0][0]
        T = input_shape[0][1]
        
        return (b, 2, T)
