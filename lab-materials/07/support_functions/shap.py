from trustyai.model import Model
from trustyai.explainers import SHAPExplainer

import pickle
import pandas as pd
import numpy as np
import keras

def shap(model_name, target, datapoint):
    keras_model = keras.saving.load_model(f"{model_name}/model.keras")    
    test_data = pd.read_parquet(f'{model_name}/X_test.parquet')
    train_data = pd.read_parquet(f'{model_name}/X_train.parquet')

    # Define the datapoint that we want to analyse, typically a strange prediction
    strange_prediction = test_data.loc[[datapoint]].drop(target, axis=1)

    def pred(x):
        prediction = keras_model.predict(x)
        return pd.DataFrame(prediction, columns=[target])

    trustyai_model = Model(pred, output_names=[target])
    prediction = trustyai_model(strange_prediction)

    # We need to make sure that all our data has the correct datatype
    for key in train_data.columns:
        test_data[[key]] = test_data[[key]].astype("float32")
        train_data[[key]] = train_data[[key]].astype("float32")
        try:
            strange_prediction[[key]] = strange_prediction[[key]].astype("float32")
        except:
            pass

    # We then pick some datapoints to be a background dataset which we will compare our datapoint to.
    # Here we make sure that the datapoints we use as background are well distributed across the dataset.
    sample_sizes = train_data[target].value_counts(normalize=True) * 100
    sample_sizes = sample_sizes.apply(lambda x: int(round(x)))
    
    correction = 100 - sample_sizes.sum()
    if correction > 0:
        sample_sizes.iloc[:correction] += 1 
    elif correction < 0:
        sample_sizes.iloc[:abs(correction)] -= 1
    
    bg_dataset = train_data.groupby(target).apply(lambda x: x.sample(n=sample_sizes[x.name], random_state=42))
    bg_dataset = bg_dataset.reset_index(drop=True).drop(target, axis=1)

    # Now we can do our SHAP analysis!
    explainer = SHAPExplainer(background=bg_dataset, seed=42)
    explanations = explainer.explain(inputs=strange_prediction,
                             outputs=prediction,
                             model=trustyai_model)

    return explanations