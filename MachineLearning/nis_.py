import numpy as np
import scipy.stats as stats

class NIS:
    def __init__(self,true_measurements, pred_measurements):

        self.seq = np.array(true_measurements) - np.array(pred_measurements)


    def get_NISLoss(self):

        K = self.seq.shape[0]  # Number of time steps
        n = self.seq.shape[1] if len(self.seq.shape) > 1 else 1  # Get the innovation dimension

        # Ensure self.seq is a 2D matrix
        self.seq = np.atleast_2d(self.seq).T  # Convert (10,) → (10, 1) to enforce column vector

        K = self.seq.shape[0]  # Number of time steps
        n = self.seq.shape[1]  # Innovation dimension

        # Initialize the sum
        NIS_sum = 0

        for k in range(K):
            nu_k = np.atleast_2d(self.seq[k])  # Ensure nu_k is at least 2D
            S_k = np.atleast_2d(self.seq[k])   # Ensure S_k is at least 2D

            # Convert S_k to a square diagonal matrix if needed
            if S_k.shape[0] != S_k.shape[1]:
                S_k = np.diag(S_k.flatten())  # Convert to a square matrix

            # Ensure S_k is invertible (add small regularization if needed)
            if np.linalg.cond(S_k) > 1e10:
                S_k += np.eye(S_k.shape[0]) * 1e-6  # Small regularization

            # Ensure nu_k is a column vector for multiplication compatibility
            nu_k = nu_k.reshape(-1, 1)  # Convert (10,) → (10,1)

            # Compute the normalized innovation squared (NIS) term
            NIS_sum += float(nu_k.T @ np.linalg.inv(S_k) @ nu_k)  # Force scalar output

        # Compute the time-averaged NIS
        NIS = (1 / K) * NIS_sum



        confidence = 0.99
        df = 1  # Change this for different measurement sizes

        alpha = 1 - confidence  # Significance level (1% for 99% confidence)

        # Get critical values from chi-square distribution
        lower_bound = stats.chi2.ppf(alpha / 2, df)  # 0.5% left tail
        upper_bound = stats.chi2.ppf(1 - alpha / 2, df)  # 99.5% right tail

        #dis_lb = abs(NIS - lower_bound)
        #dis_ub = abs(NIS - upper_bound)

        # Count number of NIS values within bounds
        within_bounds = np.logical_and(NIS >= lower_bound, NIS <= upper_bound)
        #fraction_within_bounds = np.mean(within_bounds)  # Fraction of samples in range

        # Define Reward:
        # +1 for each sample within bounds, large penalty (-10) if fraction is very low
        #reward = 10 * fraction_within_bounds - 10 * (1 - fraction_within_bounds)

        NISLoss = abs(1-within_bounds)

        return NISLoss