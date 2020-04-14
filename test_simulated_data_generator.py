from simulated_data_generator import SimulatedDataGenerator
from scipy.stats import pearsonr
import numpy as np
import unittest
import matplotlib.pyplot as plt


class TestSimulatedDataGenerator(unittest.TestCase):

    def test_correlation(self):
        number_of_variables = 256
        correlation_proportions = [0.25, 0.25, 0.5]
        correlation_edges = np.cumsum(correlation_proportions) * number_of_variables
        generator = SimulatedDataGenerator(number_of_variables,
                                           positive_correlated=correlation_proportions[0],
                                           negative_correlated=correlation_proportions[1],
                                           uncorrelated=correlation_proportions[2],
                                           missing_portion=0.5,
                                           fill_na=np.nan)

        x, missing_flag, y = generator.generate_data_logistic(1024, min_mult=0.0, max_mult=1.0)
        correlation_values = np.zeros(number_of_variables)
        for variable_idx in range(number_of_variables):
            feature_vector = x[:, variable_idx]
            correlation_value, _ = pearsonr(feature_vector, y)
            correlation_values[variable_idx] = correlation_value
            # if variable_idx < correlation_edges[0]:
            #     self.assertTrue(correlation_value > 0)
            # elif variable_idx < correlation_edges[1]:
            #     self.assertTrue(correlation_value < 0)
            # else:
            #     self.assertTrue(abs(correlation_value) < 0.2)
        plt.figure()
        plt.plot(np.arange(number_of_variables), correlation_values)
        plt.show()
        print('done')
