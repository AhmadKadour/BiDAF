# BiDAF
Implementation of the [Bi-Directional Attention Flow Model](https://arxiv.org/abs/1611.01603) (BiDAF) in Python using Keras

## Usage
You can use the provided functionality to train, predict and evaluate your custom versions of the model, by tuning the model's hyperparameters in very simple ways.

- **Setup**
  - **Initializing The Model:**

    ```python

    from BiDAF import BiDAFModel

    max_context_length = 150
    max_query_length = 15
    emdim = 100

    model = BiDAFModel(d=emdim, max_context_lenght=max_context_length,
                       max_question_lenght=max_query_length, dropout=0.4, learning_rate=0.05)

    ```
  
  - **Preparing The Data:**

    ```python

    from data_generation import get_train_data, get_validation_data

    X_train, X_test, y_train, y_test = get_train_data(max_context_length, max_query_length)
    validation_data = get_validation_data(max_context_length, max_query_length)

    ```
    
    
 - **Training**
    ```python
    
    model.run_training(X_train, y_train,
                       epochs=100, validation_data=validation_data, batch_size=64)  
    
    ```
    
    
  - **Evaluation**
    - **Model Evaluation:**
      ```python

      model.evaluate(X_test, y_test) 

      ```
    
    - **F1 and Exact Match Evaluation:**
      ```python

      from preprocess import parse_and_save_data, get_json_data
      from evaluation import get_raw_scores, make_eval_dict
      import pandas as pd
      
      json_data = get_json_data('dev')

      formated_data_path = parse_and_save_data('dev', 'dev_data')
      data_frame = pd.read_csv(formated_data_path)

      contexts = data_frame['context'].to_list()
      questions = data_frame['question'].to_list()
      qids = data_frame['qid'].to_list()

      predections_list = model.predict(contexts, questions, qids)
      predections_dict = {answer_dict['id']: answer_dict['answer'] for answer_dict in predections_list}

      em, f1 = get_raw_scores(json_data, predections_dict)
      eval_dict = make_eval_dict(em, f1, qids)

      print(eval_dict)

      ```
    
    
  - **Prediction**
    ```python
    
    context = "The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries."
    
    question = "In what country is Normandy located?"

    model.predict(context, question)
    
    ```
