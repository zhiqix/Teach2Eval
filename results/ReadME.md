# Statistical Analysis of Experimental Results  

Execute the `run.sh` script to perform statistical analysis on the experimental results.  

## Script Execution Flow  

1. **Compute Confusion Matrix**  
   - Runs `count_matrix.py`  
   - Results saved to `main_results_count`  

2. **Evaluate Model Capabilities**  
   - Runs `result_statistics.py`  
   - Results stored in:  
     - `main_results_count`  
     - `results.pkl`  
   - **Note:** Large models must be specified in `large_model_list`  

3. **Display Statistical Data**  
   - Runs `read_pickle.py` to read and display results  

## Optional Configuration  
- To adjust the number of evaluation runs (`turn` value), add the `--turn` parameter when executing the Python scripts in `run.sh`.  