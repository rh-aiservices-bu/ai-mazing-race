from trustyai.model import Model
from trustyai.model.domain import feature_domain
from trustyai.model import output
from trustyai.explainers import CounterfactualExplainer

import pickle
import pandas as pd
import numpy as np
import keras

LABEL_THRESHOLD = 0.0


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

def counterfactual(model_name, target, datapoint):
    keras_model = keras.saving.load_model(f"{model_name}/model.keras")

    with open(f'{model_name}/scalers.pkl', 'rb') as handle:
        scalers = pickle.load(handle)

    test_data = pd.read_parquet(f'{model_name}/X_test.parquet')
    strange_prediction = test_data.loc[[datapoint]].drop(target, axis=1)
    unscaled_datapoint = unscale(strange_prediction, scalers)

    def pred(x):
        x_df = pd.DataFrame(x, columns=unscaled_datapoint.columns)
        scaled_x = scale(x_df, scalers)
        prediction = keras_model.predict(scaled_x, verbose=0)
        unscaled_pred = scalers[target].inverse_transform(prediction)[0][0]
        if unscaled_pred >= LABEL_THRESHOLD:
            pred = {target: True}
        else:
            pred = {target: False}
        return pd.DataFrame([pred])

    model = Model(pred, output_names=[target])

    # Define the range of values the different features can take
    domains = {}
    for key in unscaled_datapoint.columns:
        if "category" in key or "sellable_online" in key or "other_colors" in key:
            domains[key] = feature_domain([False, True])
            unscaled_datapoint[[key]] = unscaled_datapoint[[key]].astype("bool")
        elif key in scalers.keys():
            domains[key] = feature_domain((scalers[key].data_min_[0], scalers[key].data_max_[0]))
        else:
            domains[key] = feature_domain((0, 1))
    domains = list(domains.values())

    # Set a goal for our counterfactual analysis, in our case it's to make the value possitive
    goal = [output(name=target, dtype="bool", value=True)]

    # Do the counterfactual analysis
    STEPS=50
    print("Running the Counterfactual analysis, this may take a moment...")
    explainer = CounterfactualExplainer(steps=STEPS)
    explanation = explainer.explain(inputs=unscaled_datapoint, goal=goal, model=model, feature_domains=domains)

    return explanation