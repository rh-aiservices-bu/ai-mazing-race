# !pip install -q optimum[onnxruntime] transformers torch
import time



from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoFeatureExtractor

# Load and export the model to ONNX
model = ORTModelForImageClassification.from_pretrained("rh-ai-bu/wildfire01", export=True)

# Load the feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained("rh-ai-bu/wildfire01")

# Create an ONNX-based pipeline
from optimum.pipelines import pipeline
onnx_pipeline = pipeline("image-classification", model=model, feature_extractor=feature_extractor, accelerator="ort")

onnx_pipeline.save_pretrained("onnx_pipeline")



# image = "pic2.jpg"
# Import the function from the script
from get_random_image import get_random_image

# Call the function
image = get_random_image()

# Print the selected image path
print(image)


image_preprocessed = onnx_pipeline.preprocess(image)


image_preprocessed["pixel_values"].shape

import requests

deployed_model_name = "wildfire01"
infer_endpoint = "https://wildfire01-user1-ai-mazing.apps.prod.rhoai.rh-aiservices-bu.com"
infer_url = f"{infer_endpoint}/v2/models/{deployed_model_name}/infer"
print(infer_url)

def rest_request(data):
    json_data = {
        "inputs": [
            {
                "name": "pixel_values",
                "shape": image_preprocessed["pixel_values"].shape,
                "datatype": "FP32",
                "data": data
            }
        ]
    }

    start_time = time.time()  # Record the start time
    response = requests.post(infer_url, json=json_data, verify=True)
    end_time = time.time()  # Record the end time

    # Calculate and print the response time
    response_time = end_time - start_time
    print(f"Response time: {response_time:.6f} seconds")

    return response


prediction = rest_request(image_preprocessed["pixel_values"].tolist())

prediction

prediction.json()


import numpy as np

# Extract the JSON data from the Response object
prediction_data = prediction.json()

# Extract the logits from the prediction JSON
logits = prediction_data['outputs'][0]['data']

# Define the labels corresponding to the logits
labels = [
    "Smoke_from_fires",
    "Smoke_confounding_elements",
    "Forested_areas_without_confounding_elements",
    "Both_smoke_and_fire",
    "Fire_confounding_elements"
]

# Apply the softmax function to convert logits to probabilities
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
    return exp_logits / np.sum(exp_logits)

# Get the probabilities (scores)
scores = softmax(logits)

# Combine labels with scores
results = [{'label': label, 'score': score} for label, score in zip(labels, scores)]

# Sort the results by score in descending order
results_sorted = sorted(results, key=lambda x: x['score'], reverse=True)

# Print the results
results_sorted

