

import time
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the ONNX model pipeline (assuming this is already set up as in your original code)
from optimum.onnxruntime import ORTModelForImageClassification
from transformers import AutoFeatureExtractor
from optimum.pipelines import pipeline
from get_random_image import get_random_image

# Load and export the model to ONNX
model = ORTModelForImageClassification.from_pretrained("rh-ai-bu/wildfire01", export=True)
feature_extractor = AutoFeatureExtractor.from_pretrained("rh-ai-bu/wildfire01")
onnx_pipeline = pipeline("image-classification", model=model, feature_extractor=feature_extractor, accelerator="ort")
onnx_pipeline.save_pretrained("onnx_pipeline")

# Set up the endpoint
deployed_model_name = "wildfire01"
infer_endpoint = "https://wildfire01-user1-ai-mazing.apps.prod.rhoai.rh-aiservices-bu.com"
infer_url = f"{infer_endpoint}/v2/models/{deployed_model_name}/infer"

# Function to send a single REST request
def rest_request(data):
    json_data = {
        "inputs": [
            {
                "name": "pixel_values",
                "shape": data.shape,
                "datatype": "FP32",
                "data": data.tolist()
            }
        ]
    }

    start_time = time.time()
    response = requests.post(infer_url, json=json_data, verify=True)
    end_time = time.time()

    response_time = end_time - start_time
    print(f"Response time: {response_time:.6f} seconds")

    return response

# Preprocess the image
image = get_random_image()
print(f"Selected image: {image}")
image_preprocessed = onnx_pipeline.preprocess(image)
pixel_values = image_preprocessed["pixel_values"]

# Function to handle a single prediction request
def process_request():
    response = rest_request(pixel_values)
    if response.status_code == 200:
        prediction_data = response.json()
        logits = prediction_data['outputs'][0]['data']
        scores = softmax(logits)
        return scores
    else:
        print(f"Request failed with status code: {response.status_code}")
        return None


# Softmax function
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

# Send 100 simultaneous requests using ThreadPoolExecutor
def send_simultaneous_requests():
    results = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = [executor.submit(process_request) for _ in range(100)]
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:  # Correctly check for None instead of relying on NumPy truth value
                    results.append(result)
            except Exception as e:
                print(f"Error during request processing: {e}")
    return results

# Execute the simultaneous requests
if __name__ == "__main__":
    all_results = send_simultaneous_requests()
    # print("All results:", all_results)