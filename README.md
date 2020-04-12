# Simulated Data Generator
This creates a generator that generates simulation data for testing machine learning algorithms.
It creates a labels vector (currently only for classification use so 0 and 1) and creates associated input data that
has variables that are positively correlated with it and other variables that are negatively correlated with it as well
as uncorrelated variables. Currently it only supports 1D data that can be used in the simplest ML algorithms like
fully connected neural networks. 