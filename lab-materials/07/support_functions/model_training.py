from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import utils
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import numpy as np
import pickle
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt

def pre_process_data(df, input_features, target):
    items_to_scale = ['price', 'old_price', 'width', 'height', 'depth', 'discount_amount', 'size']
    items_to_encode = ['sellable_online', 'other_colors']
    items_to_one_hot_encode = ['category']
    
    scalers = {}
    encoders = {}
    
    for i in items_to_scale:
        if i in input_features or i == target:
            scalers[i] = MinMaxScaler()
            df[[i]] = scalers[i].fit_transform(df[[i]])
        else:
            df = df.drop(i, axis=1)
    
    for i in items_to_encode:
        if i in input_features or i == target:
            encoders[i] = LabelEncoder()
            df[i] = encoders[i].fit_transform(df[i])
        else:
            df = df.drop(i, axis=1)
    
    for i in items_to_one_hot_encode:
        if i in input_features or i == target:
            df = pd.get_dummies(df, columns=[i])
        else:
            df = df.drop(i, axis=1)
    
    df = df.drop(["name", "short_description"], axis=1)
    return df, scalers, encoders

def create_model(my_learning_rate, input_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(20, activation='relu', name='Hidden1'),
        tf.keras.layers.Dense(10, activation='relu', name='Hidden2'),
        tf.keras.layers.Dense(1, name='Output')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss='mean_squared_error',
                  metrics=[tf.keras.metrics.MeanSquaredError()])
    return model

def save_model(model_name, model, df_train, df_test, scalers, encoders):
    # Save the model and artifacts
    Path(model_name).mkdir(parents=True, exist_ok=True)
    
    model.save(f"{model_name}/model.keras")
    
    with open(f"{model_name}/scalers.pkl", "wb") as handle:
        pickle.dump(scalers, handle)
    with open(f"{model_name}/encoders.pkl", "wb") as handle:
        pickle.dump(encoders, handle)
    
    df_train.to_parquet(f"{model_name}/X_train.parquet")
    df_test.to_parquet(f"{model_name}/X_test.parquet")

def train_model(df, input_features, target, split_function, model_name):
    df, scalers, encoders = pre_process_data(df, input_features, target)
    df_train, df_test = split_function(df)

    tf.keras.utils.set_random_seed(112)

    # Hyperparameters
    learning_rate = 0.01
    epochs = 20
    batch_size = 2
    
    # Create the model
    model = create_model(learning_rate, len(df_train.columns)-1)
    
    features = df_train.drop(target, axis=1)
    labels = df_train[[target]]
    
    # Fit the model
    history = model.fit(features.values, labels.values, epochs=epochs, batch_size=batch_size)

    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Training Loss')
    plt.legend()
    plt.show()

    save_model(model_name, model, df_train, df_test, scalers, encoders)
    print(f"Model and artifacts saved to folder {model_name}/")

    return df