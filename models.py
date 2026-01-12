import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
import config


def build_time_aware_model(input_dim):

    inputs = Input(shape=(input_dim,))
    
    x = BatchNormalization()(inputs)
    
    x = Dense(
        config.NN_LAYERS[0], 
        activation='relu', 
        kernel_initializer='he_normal', 
        kernel_regularizer=l2(config.L2_REGULARIZATION)
    )(x)
    x = Dropout(config.DROPOUT_RATES[0])(x)
    
    x = Dense(
        config.NN_LAYERS[1], 
        activation='relu', 
        kernel_regularizer=l2(config.L2_REGULARIZATION)
    )(x)
    x = BatchNormalization()(x)
    x = Dropout(config.DROPOUT_RATES[1])(x)
    
    x = Dense(config.NN_LAYERS[2], activation='relu')(x)
    x = Dropout(config.DROPOUT_RATES[2])(x)
    
    x = Dense(config.NN_LAYERS[3], activation='relu')(x)
    x = Dropout(config.DROPOUT_RATES[3])(x)
    
    outputs = Dense(
        1, 
        activation='sigmoid', 
        kernel_regularizer=l2(config.L2_REGULARIZATION)
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = Nadam(
        learning_rate=config.INITIAL_LEARNING_RATE, 
        clipnorm=1.0
    )
    
    model.compile(
        loss='mse', 
        optimizer=optimizer, 
        metrics=['mae']
    )
    
    return model


def build_model_ensemble(input_dim, n_models=None):

    if n_models is None:
        n_models = config.ENSEMBLE_SIZE
    
    models = []
    for _ in range(n_models):
        models.append(build_time_aware_model(input_dim))
    
    return models