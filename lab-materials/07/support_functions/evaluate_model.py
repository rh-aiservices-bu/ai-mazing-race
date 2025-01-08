import pickle
import pandas as pd
import numpy as np
import keras

def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df

def mse(model, test_features, test_labels, target, scalers):    
    print("\n Evaluate the new model against the test set:")
    result = model.evaluate(x = test_features, y = test_labels)[1]
    unscaled_result = scalers[target].inverse_transform([[result]])[0][0]
    print(f"The model has an MSE of {unscaled_result} ({result} when scaled) on the test data.")

def below_zero(model, test_data, test_features, test_labels, target, scalers):
    # Get the predictions
    prediction = model(test_features)
    predicted_df = test_data.copy()
    predicted_df[[target]] = prediction

    #Unscale everything
    for k, v in scalers.items():
        if k in predicted_df.columns:
            predicted_df[[k]] = v.inverse_transform(predicted_df[[k]])

    # Reverse the one-hot encoding/dummyfying
    categores = [col for col in predicted_df.columns if "category" in col]
    if categores:
        concat_categories = undummify(predicted_df[categores])
        predicted_df = predicted_df.drop(categores, axis=1)
        predicted_df["category"] = concat_categories

    return predicted_df, predicted_df[predicted_df["price"]<0]

def worst_prediction(predicted_df, test_labels, target, scalers):
    unscaled_labels = scalers[target].inverse_transform(test_labels)
    worst_prediction = np.argmax(np.abs(predicted_df[target].values.reshape(-1, 1) - unscaled_labels))
    wort_pred_id = predicted_df.index[worst_prediction]
    print(f"Worst prediction has id: {wort_pred_id}")

    comparison = {
        "prediction": predicted_df.reset_index().iloc[worst_prediction][target],
        "target": unscaled_labels[worst_prediction][0],
        "error": abs(predicted_df.reset_index().iloc[worst_prediction][target] - unscaled_labels[worst_prediction][0])
    }
    return pd.DataFrame([comparison], index=[wort_pred_id])

def model_evaluation(model_name, target):
    model = keras.saving.load_model(f"{model_name}/model.keras")

    test_data = pd.read_parquet(f'{model_name}/X_test.parquet')
    
    with open(f'{model_name}/scalers.pkl', 'rb') as handle:
        scalers = pickle.load(handle)

    test_features = test_data.drop(target, axis=1)
    test_labels = test_data[[target]]

    mse(model, test_features, test_labels, target, scalers)

    predicted_df, below_zero_df = below_zero(model, test_data, test_features, test_labels, target, scalers)

    worst_df = worst_prediction(predicted_df, test_labels, target, scalers)

    return worst_df, below_zero_df