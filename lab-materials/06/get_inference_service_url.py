import subprocess

def run_command(command):
    """Run an OS command and return its output, handling errors."""
    try:
        result = subprocess.check_output(command, shell=True, text=True).strip()
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the command '{command}': {e}")
        return None

def get_inference_service_url():
    """Retrieve the full inference service URL and model name."""
    # Commands to run
    get_IS_name = "oc get inferenceservice -o jsonpath='{.items[0].metadata.name}'"
    get_IS_URL = "oc get inferenceservice -o jsonpath='{.items[0].status.url}'"

    # Retrieve results
    deployed_model_name = run_command(get_IS_name)
    infer_endpoint = run_command(get_IS_URL)

    if not deployed_model_name:
        print("Failed to retrieve Inference Service Name.")
        return None

    if not infer_endpoint:
        print("Failed to retrieve Inference Service URL.")
        return None

    infer_url = f"{infer_endpoint}/v2/models/{deployed_model_name}/infer"
    print("Inference Service Name:", deployed_model_name)
    print("Inference Service URL:", infer_endpoint)
    print("Full URL:", infer_url)

    return infer_url

# Example usage (for a notebook, you can call this function)
if __name__ == "__main__":
    url = get_inference_service_url()
    if url:
        print("Generated Inference URL:", url)
