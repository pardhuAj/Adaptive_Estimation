import pandas as pd
import matplotlib.pyplot as plt

# Load simulation results
df = pd.read_csv("simulation_results.csv")

# Plot Q_true vs. Q_hat
plt.figure(figsize=(10, 5))
plt.plot(df["Q_true"], label="Q_true", linestyle="dashed", color="blue", alpha=0.7)
plt.plot(df["Q_hat"], label="Q_hat", color="red", alpha=0.8)
plt.xlabel("Timestep")
plt.ylabel("Process Noise Q")
plt.legend()
plt.title("Estimated vs. True Process Noise (Q)")
plt.grid()
plt.show()

# Plot R_true vs. R_hat
plt.figure(figsize=(10, 5))
plt.plot(df["R_true"], label="R_true", linestyle="dashed", color="blue", alpha=0.7)
plt.plot(df["R_hat"], label="R_hat", color="red", alpha=0.8)
plt.xlabel("Timestep")
plt.ylabel("Measurement Noise R")
plt.legend()
plt.title("Estimated vs. True Measurement Noise (R)")
plt.grid()
plt.show()

# Plot x_true_1 vs. x_hat_1
plt.figure(figsize=(10, 5))
plt.plot(df["x_true_1"], label="x_true_1 (True State)", linestyle="dashed", color="blue", alpha=0.7)
plt.plot(df["x_hat_1"], label="x_hat_1 (Estimated State)", color="red", alpha=0.8)
plt.xlabel("Timestep")
plt.ylabel("State x1")
plt.legend()
plt.title("State Estimate Comparison (x1)")
plt.grid()
plt.show()

# Plot Error state 1
plt.figure(figsize=(10, 5))
plt.plot(df["x_true_1"] - df["x_hat_1"], label="Error state 1", linestyle="dashed", color="red", alpha=0.7)
#plt.plot(, label="x_hat_1 (Estimated State)", color="red", alpha=0.8)
plt.plot(3*df["P1_updated"], label="P1_updated", linestyle="dashed", color="blue", alpha=0.7)
plt.plot(-3*df["P1_updated"], label="P1_updated", linestyle="dashed", color="blue", alpha=0.7)
plt.xlabel("Timestep")
plt.ylabel("State x1")
plt.legend()
plt.title("State Estimate Comparison (x1)")
plt.grid()
plt.show()

# Plot x_true_2 vs. x_hat_2
plt.figure(figsize=(10, 5))
plt.plot(df["x_true_2"], label="x_true_2 (True State)", linestyle="dashed", color="blue", alpha=0.7)
plt.plot(df["x_hat_2"], label="x_hat_2 (Estimated State)", color="red", alpha=0.8)
plt.xlabel("Timestep")
plt.ylabel("State x2")
plt.legend()
plt.title("State Estimate Comparison (x2)")
plt.grid()
plt.show()

# Error 2
plt.figure(figsize=(10, 5))
plt.plot(df["x_true_2"] - df["x_hat_2"], label="Error state 2", linestyle="dashed", color="red", alpha=0.7)
#plt.plot(df["x_hat_2"], label="x_hat_2 (Estimated State)", color="red", alpha=0.8)
plt.plot(3*df["P2_updated"], label="P2_updated", linestyle="dashed", color="blue", alpha=0.7)
plt.plot(-3*df["P2_updated"], label="P2_updated", linestyle="dashed", color="blue", alpha=0.7)
plt.xlabel("Timestep")
plt.ylabel("State x2")
plt.legend()
plt.title("State Estimate Comparison (x2)")
plt.grid()
plt.show()
