# Testing Model Standalone Performance  

To test the models' standalone performance, run the `run.sh` script.  

## Prerequisites  

### 1. Configure `run.sh`  
- Specify the list of **models** to evaluate  

### 2. Download Required Models  
- Ensure all necessary models are downloaded  
- If you are using an online API model, please fill in the model name, base_url, and api in config.json.

### 3. Register Model Paths  
- Update `code/model.py` following the provided example  
- Register all required models in this file  