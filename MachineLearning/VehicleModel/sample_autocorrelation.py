import numpy as np

class SampleAutocorrelation:
    def __init__(self, true_measurements, pred_measurements):
        self.N = len(true_measurements)
        self.M = self.N // 2
        self.seq = np.asarray(true_measurements).flatten() - np.asarray(pred_measurements).flatten()

    def get_J(self):
        M = int(self.M)
        N = int(self.N)
        k = np.zeros(N - M)
        C_est = np.zeros(M)

        C_0 = (np.sum(self.seq[:len(self.seq) // 2] ** 2)) / (N - M)
        for i in range(1, M):
            for j in range(1, N - M):
                k[j - 1] = float(self.seq[j]) * float(self.seq[j + i])
            C_est[i] = np.sum(k) / (N - M)

        C_0_diag_matrix = np.array([[C_0]]) if np.isscalar(C_0) else np.diag(C_0)
        C_0_diag_inv_sqrt = np.linalg.inv(C_0_diag_matrix) ** 0.5
        C_0_diag_inv = np.linalg.inv(C_0_diag_matrix)

        J_sum = 0
        for i in range(1, M):
            C_i = np.atleast_2d(C_est[i])
            term1 = C_0_diag_inv_sqrt @ C_i.T
            term2 = C_0_diag_inv @ C_i @ C_0_diag_inv_sqrt
            J_sum += np.trace(term1 @ term2)

        return 0.5 * J_sum
