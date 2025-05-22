Execute the run.sh script to statistically analyze the experimental results.

The script will sequentially:

1. Run count_matrix.py to compute the confusion matrix, with results saved to main_results_count.
2. Run result_statistics.py to evaluate various capabilities, with results stored in main_results_count and results.pkl (Note: Large models to be evaluated must be specified in large_model_list).
3. Run read_pickle.py to read and display the statistical data.

If you need to adjust the turn value for the runs, you can add the --turn parameter when executing the Python scripts in run.sh.