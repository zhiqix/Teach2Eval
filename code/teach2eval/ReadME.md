# Running the Teach2Eval Program

To execute the **Teach2Eval** main program, run the `run.sh` script.

## Prerequisites

### 1. Configure `run.sh`
- Specify the list of **large models** to evaluate
- Specify the list of **small models** to evaluate
- Set the `--turn` value (number of teaching turns)

### 2. Download Required Models
- Ensure all necessary models are downloaded before execution
- If you are using an online API model, please fill in the model name, base_url, and api in config.json.

### 3. Register Model
- Update `code/model.py` following the provided example
- Register all required models in this file
  
### 4. Pre-test Large Models
- Before running `run.sh`, test the large models' standalone performance
- Execute `code/test_model/run.sh` to perform this pre-testing