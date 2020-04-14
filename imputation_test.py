from fancyimpute import KNN, IterativeImputer, IterativeSVD, MatrixFactorization, SoftImpute, BiScaler, SimpleFill
from simulated_data_generator import SimulatedDataGenerator
import numpy as np

"""Testing imputation using fancyimpute library"""

# X is the complete data matrix
number_of_variables = 256
correlation_proportions = [0.25, 0.25, 0.5]
correlation_edges = np.cumsum(correlation_proportions) * number_of_variables
generator = SimulatedDataGenerator(number_of_variables,
                                   positive_correlated=correlation_proportions[0],
                                   negative_correlated=correlation_proportions[1],
                                   uncorrelated=correlation_proportions[2],
                                   missing_portion=0.0,
                                   fill_na=np.nan)

X, _, y = generator.generate_data_logistic(1024, min_mult=0.0, max_mult=1.0)
# X_incomplete has the same values as X except a subset have been replace with NaN
X_incomplete, missing_mask = generator.generate_missing(X, 0.1, np.nan)

# Use 3 nearest rows which have a feature to fill in each row's missing features
X_filled_knn = KNN(k=3).fit_transform(X_incomplete)

# matrix completion using MICE
X_filled_mice = IterativeImputer().fit_transform(X_incomplete)

# matrix completion using Iterative SVD
X_filled_svd = IterativeSVD(rank=3).fit_transform(X_incomplete)

# matrix completion using Matrix Factorization
X_filled_mf = MatrixFactorization(learning_rate=0.01,
                                  rank=3,
                                  l2_penalty=0,
                                  min_improvement=1e-6).fit_transform(X_incomplete)

# matrix completion using Mean Fill
X_filled_meanfill = SimpleFill(fill_method='mean').fit_transform(X_incomplete)
# matrix completion using Median Fill
X_filled_medianfill = SimpleFill(fill_method='median').fit_transform(X_incomplete)
# matrix completion using Zero Fill
X_filled_zerofill = SimpleFill(fill_method='zero').fit_transform(X_incomplete)
# matrix completion using Min Fill
X_filled_minfill = SimpleFill(fill_method='min').fit_transform(X_incomplete)
# matrix completion using Sampled Fill
X_filled_randomfill = SimpleFill(fill_method='random').fit_transform(X_incomplete)

# Instead of solving the nuclear norm objective directly, instead
# induce sparsity using singular value thresholding
X_incomplete_normalized = BiScaler().fit_transform(X_incomplete)
X_filled_softimpute = SoftImpute().fit_transform(X_incomplete_normalized)

# print mean squared error for the  imputation methods above
mice_mse = ((X_filled_mice[missing_mask] - X[missing_mask]) ** 2).mean()
print("MICE MSE: %f" % mice_mse)

svd_mse = ((X_filled_svd[missing_mask] - X[missing_mask]) ** 2).mean()
print("SVD MSE: %f" % svd_mse)

mf_mse = ((X_filled_mf[missing_mask] - X[missing_mask]) ** 2).mean()
print("Matrix Factorization MSE: %f" % mf_mse)

meanfill_mse = ((X_filled_meanfill[missing_mask] - X[missing_mask]) ** 2).mean()
print("MeanImpute MSE: %f" % meanfill_mse)

medianfill_mse = ((X_filled_medianfill[missing_mask] - X[missing_mask]) ** 2).mean()
print("MedianImpute MSE: %f" % medianfill_mse)

zerofill_mse = ((X_filled_zerofill[missing_mask] - X[missing_mask]) ** 2).mean()
print("ZeroImpute MSE: %f" % zerofill_mse)

minfill_mse = ((X_filled_minfill[missing_mask] - X[missing_mask]) ** 2).mean()
print("MinImpute MSE: %f" % minfill_mse)

randomfill_mse = ((X_filled_randomfill[missing_mask] - X[missing_mask]) ** 2).mean()
print("SampledImpute MSE: %f" % randomfill_mse)

softImpute_mse = ((X_filled_softimpute[missing_mask] - X[missing_mask]) ** 2).mean()
print("SoftImpute MSE: %f" % softImpute_mse)

knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
print("knnImpute MSE: %f" % knn_mse)