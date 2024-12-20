from trustyai.model import Model
from trustyai.model.domain import feature_domain
from trustyai.model import output
from trustyai.explainers import CounterfactualExplainer

import pickle
import pandas as pd
import numpy as np
import keras

LABEL_THRESHOLD = 0.0

def counterfactual(model_name, target, datapoint):
    keras_model = keras.saving.load_model(f"{model_name}/model.keras")

    with open(f'{model_name}/scalers.pkl', 'rb') as handle:
        scalers = pickle.load(handle)

    test_data = pd.read_parquet(f'{model_name}/X_test.parquet')
    strange_prediction = test_data.loc[[datapoint]].drop(target, axis=1)
    strange_prediction

    def pred(x):
        prediction = keras_model.predict(x)
        unscaled_pred = scalers[target].inverse_transform(prediction)[0][0]
        print(unscaled_pred)
        if unscaled_pred >= LABEL_THRESHOLD:
            pred = {target: True}
        else:
            pred = {target: False}
        return pd.DataFrame([pred])

    model = Model(pred, output_names=[target])

    # Define the range of values the different features can take
    domains = {}
    for key in strange_prediction.columns:
        if "category" in key or "sellable_online" in key or "other_colors" in key:
            domains[key] = feature_domain([False, True])
            strange_prediction[[key]] = strange_prediction[[key]].astype("bool")
        else:
            domains[key] = feature_domain((0, 1))
    domains = list(domains.values())

    # Set a goal for our counterfactual analysis, in our case it's to make the value possitive
    goal = [output(name=target, dtype="bool", value=True)]

    # Do the counterfactual analysis
    STEPS=50
    explainer = CounterfactualExplainer(steps=STEPS)
    explanation = explainer.explain(inputs=strange_prediction, goal=goal, model=model, feature_domains=domains)

    return explanation