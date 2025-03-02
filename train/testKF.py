import numpy as np
import matplotlib.pyplot as plt

from kalmanFilter import KalmanFilter2D

if __name__ == "__main__":
    # Simulation parameters
    dt = 1.0  # Time step
    num_steps = 30  # Number of time steps

    # True position and velocity (assumed constant velocity)
    true_position = 0
    true_velocity = 1

    # Initial state [position, velocity]
    x0 = np.array([0, 0])

    # Initial uncertainty
    P0 = np.array([[1, 0], [0, 1]])

    # State transition model (Constant velocity model)
    F = np.array([[1, dt], [0, 1]])

    # Measurement model (we only measure position)
    H = np.array([[1, 0]])

    # Process noise covariance (assumed small)
    Q = np.array([[0.01, 0], [0, 0.01]])

    # Measurement noise covariance (assumed moderate)
    R = np.array([[0.5]])

    kf = KalmanFilter2D(x0, P0, F, H, Q, R)

    # Data storage
    true_positions = []
    measurements = []
    estimated_positions = []
    estimated_velocities = []

    # Simulate noisy measurements and Kalman Filter estimation
    for i in range(num_steps):
        true_position += true_velocity * dt  # Update true position
        measurement = true_position + np.random.normal(0, np.sqrt(R[0, 0]))  # Add measurement noise

        kf.predict()
        kf.update(np.array([measurement]))

        # Store data for plotting
        true_positions.append(true_position)
        measurements.append(measurement)
        estimated_positions.append(kf.x[0])
        estimated_velocities.append(kf.x[1])
        print(kf.S)
        print(kf.y)

    # Plot Results
    plt.figure(figsize=(12, 5))

    # Position plot
    plt.subplot(1, 2, 1)
    plt.plot(true_positions, label="True Position", linestyle='dashed', color='black')
    plt.scatter(range(num_steps), measurements, label="Noisy Measurements", color='red', alpha=0.5)
    plt.plot(estimated_positions, label="Kalman Filter Estimate", color='blue')
    plt.xlabel("Time Step")
    plt.ylabel("Position")
    plt.title("Position Tracking with Kalman Filter")
    plt.legend()
    plt.grid()

    # Velocity plot
    plt.subplot(1, 2, 2)
    plt.plot(estimated_velocities, label="Estimated Velocity", color='green')
    plt.axhline(true_velocity, color='black', linestyle='dashed', label="True Velocity")
    plt.xlabel("Time Step")
    plt.ylabel("Velocity")
    plt.title("Velocity Estimation with Kalman Filter")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
