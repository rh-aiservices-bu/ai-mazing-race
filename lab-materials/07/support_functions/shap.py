from trustyai.model import Model
from trustyai.explainers import SHAPExplainer

import pickle
import pandas as pd
import numpy as np
import keras

def scale(datapoint, scalers):
    scaled_datapoint = datapoint.copy()
    for k, v in scalers.items():
        if k in datapoint.columns:
            scaled_datapoint[[k]] = v.transform(datapoint[[k]])
    return scaled_datapoint

def unscale(datapoint, scalers):
    unscaled_datapoint = datapoint.copy()
    for k, v in scalers.items():
        if k in datapoint.columns:
            unscaled_datapoint[[k]] = v.inverse_transform(datapoint[[k]])
    return unscaled_datapoint

def shap(model_name, target, datapoint):
    keras_model = keras.saving.load_model(f"{model_name}/model.keras")    
    test_data = pd.read_parquet(f'{model_name}/X_test.parquet')
    train_data = pd.read_parquet(f'{model_name}/X_train.parquet')
    with open(f'{model_name}/scalers.pkl', 'rb') as handle:
        scalers = pickle.load(handle)

    unscaled_test_data = unscale(test_data, scalers)
    unscaled_train_data = unscale(train_data, scalers)

    # Define the datapoint that we want to analyse, typically a strange prediction
    strange_prediction = unscaled_test_data.loc[[datapoint]].drop(target, axis=1)

    def pred(x):
        x_df = pd.DataFrame(x, columns=strange_prediction.columns)
        scaled_x = scale(x_df, scalers)
        prediction = keras_model.predict(scaled_x, verbose=0)
        unscaled_pred = scalers[target].inverse_transform(prediction)
        return pd.DataFrame(unscaled_pred, columns=[target])

    trustyai_model = Model(pred, output_names=[target])
    prediction = trustyai_model(strange_prediction)

    # We need to make sure that all our data has the correct datatype
    for key in unscaled_train_data.columns:
        unscaled_test_data[[key]] = unscaled_test_data[[key]].astype("float32")
        unscaled_train_data[[key]] = unscaled_train_data[[key]].astype("float32")
        try:
            strange_prediction[[key]] = strange_prediction[[key]].astype("float32")
        except:
            pass

    # We then pick some datapoints to be a background dataset which we will compare our datapoint to.
    # Here we make sure that the datapoints we use as background are well distributed across the dataset.
    sample_sizes = unscaled_train_data[target].value_counts(normalize=True) * 100
    sample_sizes = sample_sizes.apply(lambda x: int(round(x)))
    
    correction = 100 - sample_sizes.sum()
    if correction > 0:
        sample_sizes.iloc[:correction] += 1 
    elif correction < 0:
        sample_sizes.iloc[:abs(correction)] -= 1
    
    bg_dataset = unscaled_train_data.groupby(target).apply(lambda x: x.sample(n=sample_sizes[x.name], random_state=42))
    bg_dataset = bg_dataset.reset_index(drop=True).drop(target, axis=1)

    # Now we can do our SHAP analysis!
    print("Running the SHAP analysis, this may take a moment...")
    explainer = SHAPExplainer(background=bg_dataset, seed=42)
    explanations = explainer.explain(inputs=strange_prediction,
                             outputs=prediction,
                             model=trustyai_model)

    return explanations