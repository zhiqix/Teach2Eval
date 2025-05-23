# You should register the model you want to use here, including the model file save path, how many GPUs are needed, and whether it is an online API call to the model.
# For models that require an online API call, there is no need to fill in model_path and model_gpu_use.
model_path = {
    "<large_model_name>": "<large_model_path>",
    "<small_model_name>": "<small_model_path>",
}

model_gpu_use = {
    "<large_model_name>": 4,
    "<small_model_name>": 1,
}

model_use_api = {
    "<large_model_name>": True,
}