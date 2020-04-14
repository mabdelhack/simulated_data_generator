import numpy as np


class SimulatedDataGenerator(object):
    """This class creates simulation data"""
    def __init__(self, sample_shape, missing_portion=0, noise_variance=1, fill_na=np.nan,
                 positive_correlated=0.25, negative_correlated=0.25, uncorrelated=0.5):
        assert positive_correlated + negative_correlated + uncorrelated == 1.0
        self.sample_shape = sample_shape
        self.missing_portion = missing_portion
        self.noise_variance = noise_variance
        self.fill_na = fill_na
        self.data_proportions = [positive_correlated, negative_correlated, uncorrelated]

    def generate_data_logistic(self, number_of_samples, proportion_positive=0.5, min_mult=0.2, max_mult=1.0):
        """Generates data that would be used for a classification task where the target is either 1 or 0"""
        label = np.random.rand(number_of_samples)
        y = label < proportion_positive
        y = y.astype(int)
        x = np.random.normal(loc=0.0, scale=self.noise_variance, size=(number_of_samples, self.sample_shape))
        data_edges = np.cumsum(self.data_proportions)

        # positively correlated
        x[:, :int(data_edges[0] * self.sample_shape)] = x[:, :int(data_edges[0]*self.sample_shape)] + \
            np.matmul(np.expand_dims(y, axis=1),
                      np.abs(np.random.uniform(low=min_mult, high=max_mult,
                                               size=(1, int(self.sample_shape*self.data_proportions[0]))
                                               )
                             )
                      )
        # negatively correlated
        x[:, int(data_edges[0] * self.sample_shape):int(data_edges[1] * self.sample_shape)] = \
            x[:, int(data_edges[0] * self.sample_shape):int(data_edges[1] * self.sample_shape)] - \
            np.matmul(np.expand_dims(y, axis=1),
                      np.abs(np.random.uniform(low=min_mult, high=max_mult,
                                               size=(1, int(self.sample_shape*self.data_proportions[1]))
                                               )
                             )
                      )
        missing_flag = np.zeros((number_of_samples, self.sample_shape))
        if self.missing_portion > 0:
            missing_probability = np.random.uniform(low=0.0, high=1.0, size=missing_flag.shape)
            missing_flag = missing_probability < self.missing_portion
            missing_flag = missing_flag.astype(int)
        x = np.ma.array(x, mask=missing_flag)
        x = x.filled(self.fill_na)
        # shuffling samples
        sample_order = np.arange(number_of_samples)
        np.random.shuffle(sample_order)
        x = x[sample_order]
        missing_flag = missing_flag[sample_order]
        y = y[sample_order]
        return x, missing_flag, y

    @staticmethod
    def generate_missing(x, missing_portion, fill_na):
        """This generates a version of data already generated with missing values"""
        missing_flag = np.zeros(x.shape)
        if missing_portion > 0:
            missing_probability = np.random.uniform(low=0.0, high=1.0, size=x.shape)
            missing_flag = missing_probability < missing_portion
            missing_flag = missing_flag.astype(int)
        x = np.ma.array(x, mask=missing_flag)
        x = x.filled(fill_na)
        return x, missing_flag
