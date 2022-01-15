# -*- coding: utf-8 -*-


from BiDAF import BiDAFModel
from data_generation import get_train_data, get_validation_data


max_context_length = 150
max_query_length = 15
emdim = 100

X_train, X_test, y_train, y_test = get_train_data(max_context_length, max_query_length)
validation_data = get_validation_data(max_context_length, max_query_length)

model = BiDAFModel(emdim, max_context_length, max_query_length, 0.4, 0.05)

model.run_training(X_train, y_train, 100, validation_data, 64)