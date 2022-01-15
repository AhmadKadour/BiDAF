# -*- coding: utf-8 -*-

import pandas as pd
import os
import nltk

nltk.download('punkt')

DIR_BASE = os.path.dirname(__file__)
DIR_DATA = os.path.join(DIR_BASE, "data")

PATH_DATA_JSON = os.path.join(DIR_DATA, "squad", "{}-v2.0.json")



def clean_sequence(secquence):
    """
    cleaning the data as recomended by the original BiDAF paper
    """
            
    secquence = secquence.replace("''", '" ')
    secquence = secquence.replace("``", '" ')
    return secquence

def clean_token(token):    
    token = token.replace("``", '"')
    token = token.replace("''", '"')
    token = token.lower()
    return token


def tokenize(sentence):
    sentence = clean_sequence(sentence)

    tokens = [clean_token(token) for token in nltk.word_tokenize(sentence)]
    return tokens

def get_token_index_dict(context, tokens):
    """
    for mapping the span of the answer
    from character based indexing on context
    to tokens based indexing on tokens

    Parameters
    ----------
    context : str
        the original context text.
    tokens : list of str
        the tokens of the context.

    Returns
    -------
    dictionary form int (the character index) to int (the token index).
    or None if an error occurred
    """
    context = context.lower()
        
    current_context_char_idx = 0
    
    mapping = {}
    
    try:
        for i, token in enumerate(tokens):
            accumulator = ""
            current_token_char_idx = 0
            
            while (accumulator != token
                   and current_token_char_idx < len(token)
                   and current_context_char_idx < len(context)):
                
                if token[current_token_char_idx] == context[current_context_char_idx]:
                    accumulator += context[current_context_char_idx]
                    current_token_char_idx += 1
                    
                mapping[current_context_char_idx] = i
                current_context_char_idx += 1
                
            if accumulator != token or accumulator == "": return None
        
        return mapping
    
    except:
        return None
            

def parse_data(data_type):
    """
    Parameters
    ----------
    data_type : str
        the type of the data ("train" or "dev")
    
    Returns
    -------
    the processed data as Pandas DataFrame
    """
    
    data_json = pd.read_json(PATH_DATA_JSON.format(data_type))
    
    contexts = []
    contexts_tokens = []
    questions = []
    qids = []
    questions_tokens = []
    is_impossible = []
    answers_text = []
    answers_char_idx_span = []
    answers_token_idx_span = []
    
    for i in range(data_json.shape[0]):
        paragraphs = data_json.iloc[i,1]['paragraphs']
        
        for paragraph in paragraphs:
            
            context = paragraph['context']
            
            context_tokens = tokenize(context)
            
            context_token_idx_mapping = get_token_index_dict(context, context_tokens)
            if context_token_idx_mapping is None: continue
        
            for q_a in paragraph['qas']:
                qid = q_a['id']
                question = q_a['question'].strip()
                question_tokens = tokenize(question)
                
                is_impossible_question = q_a['is_impossible']
                
                if is_impossible_question:
                    if not q_a['plausible_answers']:
                        answer_text = None
                        answer_start_idx = None
                        answer_end_idx = None
                    else:
                        answer_text = q_a['plausible_answers'][0]['text']
                        answer_start_idx = q_a['plausible_answers'][0]['answer_start']
                        answer_end_idx = answer_start_idx + len(answer_text)
                else:
                    answer_text = q_a['answers'][0]['text']
                    answer_start_idx = q_a['answers'][0]['answer_start']
                    answer_end_idx = answer_start_idx + len(answer_text)
                
                answer_char_idx_span = (answer_start_idx, answer_end_idx) if (not is_impossible_question) else None
                
                answer_token_idx_span = (context_token_idx_mapping[answer_start_idx],
                                         context_token_idx_mapping[answer_end_idx - 1] + 1) if (not is_impossible_question) else None

                answer_tokens = context_tokens[answer_token_idx_span[0]:answer_token_idx_span[1]] if (not is_impossible_question) else None
                
                if not is_impossible_question:
                    if context[answer_start_idx:answer_end_idx] != answer_text:
                        continue
                    if "".join(answer_tokens) != "".join(tokenize(answer_text)):
                        continue
                    
                contexts.append(context)
                contexts_tokens.append(context_tokens)
                questions.append(question)
                qids.append(qid)
                questions_tokens.append(question_tokens)
                answers_text.append(answer_text)
                answers_char_idx_span.append(answer_char_idx_span)
                answers_token_idx_span.append(answer_token_idx_span)
                is_impossible.append(is_impossible_question)
                
    return pd.DataFrame({"context": contexts,
                         "context_tokens": contexts_tokens,
                         "question": questions,
                         "qid": qids,
                         "question_tokens": questions_tokens,
                         "answer_text": answers_text,
                         "answer_char_idx_span": answers_char_idx_span,
                         "answer_token_idx_span": answers_token_idx_span,
                         "is_impossible": is_impossible})


def parse_and_save_data(data_type, file_name, file_dir = DIR_DATA):
    """
    parsing the data from the json file and the saving the results in provided directory

    Parameters
    ----------
    data_type : str
        the type of the data ("train" or "dev").
    file_name : str
        the name for the data file to be saved with.
    file_dir : str, optional
        the directory for the data file to be saved in.

    Returns
    -------
    the file path for the saved data in csv format.

    """
    
    file_path = os.path.join(file_dir, file_name + ".csv")
    if not os.path.exists(file_path):
        data = parse_data(data_type)
        data.to_csv(file_path)
    return file_path


def get_json_data(data_type):
    """
    getting raw json data as pandas DataFrame.

    Parameters
    ----------
    data_type : str
        the type of the data ("train" or "dev").
        
    """

    return pd.read_json(PATH_DATA_JSON.format(data_type))
    