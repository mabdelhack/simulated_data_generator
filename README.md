# Simulated Data Generator
This creates a generator that generates simulation data for testing machine learning algorithms.
It creates a labels vector (now supports classification and linear data) and creates associated input data that
has variables that are positively correlated with it and other variables that are negatively correlated with it as well
as uncorrelated variables. Currently it only supports 1D data that can be used in the simplest ML algorithms like
fully connected neural networks. 

## Usage
You can find usage in the unit test and the associated python notebook.
Below are the correlation results of creating 1024 samples of 256 variables of which first 64 are positively correlated, second 64 are negatively correlated and the remaining 128 are uncorrelated.



![](./images/variable_sample.png)

## Imputation
imputation_test.py tests methods from the library [FancyImpute](https://pypi.org/project/fancyimpute/) to restore the missing values introduced to the data and compares it to the real values.
