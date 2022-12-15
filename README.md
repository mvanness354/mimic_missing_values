# Supervised Learning on Mimic with Missing Values

This is the code to accompany my final project for BIODS 220 Fall 2022. 
In this project, I access the performance of various supervised learning models for handling EHR data with missing values.
IMPORTANT: this github does not contain the actual data needed to run the models, as it is too large, please see the next section for how to generate the data.
The scripts necessary to reproduce the results are in the `run_scripts/` directory.
The modules necessary to run the scripts can be found in `requirements.txt`, which can be imported in a virtual environment using
`pip install -r requirements.txt`. For a conda environment, first install pip in conda and then import the requirements using pip.

## Generating Data

The data for this project comes from [MIMIC-III](https://physionet.org/content/mimiciii/1.4/), and more specifically, the [mimic3-benchmark](https://github.com/YerevaNN/mimic3-benchmarks). Use the following steps to generate the data used to run the models in this repo:

1) Download the raw MIMIC-III data csvs from physionet, and store them in some directory. You will need to gain access to the raw data by completing the required training if not done already.
2) Follow the steps in the mimic3-benchmark README through the "Train / validation split" step. This will generate train, validation, and test data for the 4 clinical tasks, but in raw panel data form. Note that this whole process will take several hours to get through all the steps.
3) There are scripts named `mimic/make_processed_{taskname}_data.csv` to generate the tabular datasets from the raw time series data for each of the 4 tasks. Run each of these scripts to generate the corresponding csv files for each task.
