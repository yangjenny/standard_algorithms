# Algorithms used in UK-Vietnam machine learning generalizability study

This repository hosts the version of the code used for the publication ["Generalizability Assessment of AI Models Across Hospitals: A Comparative Study in Low-Middle Income and High Income Countries"]([https://www.medrxiv.org/content/10.1101/2023.11.05.23298109v1]). 

## Dependencies

Experiments were performed using Python (v3.8.3). All models were run using an Intel Xeon E-2146G Processor (CPU: 6 cores, 4.50 GHz max frequency). Logistic regression was implemented using Scikit-learn (v1.3.2). The XGBoost classifier was implemented using the XGBoost package (v2.0.3). Neural network models were implemented using Pytorch (2.1.2+cu121).

To use this branch, you can run the following lines of code:

```
conda create -n StandardAlgoEnv python==3.8
conda activate StandardAlgoEnv
git clone https://github.com/yangjenny/standard_algorithms.git
cd StandardAlgoEnv
pip install -e .
```

## Getting Started

To run code: 

```
python run.py
```

(UCI Adult dataset automatically loaded for training)

This example uses the UCI Adult dataset, where one is trying to classify income (two classes: <=50K and >50K), and mitigate gender (male vs female) bias. Additional details about the dataset, including all attributes included, can be found [here](https://archive.ics.uci.edu/ml/datasets/Adult).

An example run and expected output can be found in algorithms.ipynb

## Citation

If you found our work useful, please consider citing:

Yang, J., Dung, N. T., Thach, P. N., Phong, N. T., Phu, V. D., Phu, K. D., ... & Clifton, D. A. (2023). Generalizability assessment of AI models across hospitals: a comparative study in low-middle income and high income countries. medRxiv, 2023-11.
