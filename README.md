This is the github repository for **Sub-Projects 1 and 2**.

All results from the last run are stored in the results folder. Inside it, you'll find:
- optimizer_comparison_summary.csv: file that contains results from the last run for all optimizers (custom and TF-based ones) across all datasest and network architectures)
- plots/ directory:
  - All the training loss convergence plots for each optimizer, dataset and network architecture
  - custom_bar_plots/ and tf_bar_plots/ directories containing the per optimizer validation loss and task-specific metric bar plots used in the research paper
  
To test the code all you need to do is run the **optimizer_comparison.py** script. 

**Important:** To run the code properly, refer to the requirements.txt file for the project dependencies.

