import numpy as np
from kalman_filter_control import KalmanFilterWithControl  # Import the class

if __name__ == "__main__":
    dt = 1.0  # Time step
    num_steps = 30  # Number of time steps

    # Initial state [position, velocity]
    x0 = np.array([0, 0])

    # Initial uncertainty
    P0 = np.array([[1, 0], [0, 1]])

    # Process noise covariance (assumed small)
    Q = np.array([[0.01, 0], [0, 0.01]])

    # Measurement noise covariance (assumed moderate)
    R = np.array([[0.5]])

    # Control input (constant acceleration)
    u = np.array([0.1])  # Assume constant acceleration of 0.1 m/s²

    # Create Kalman Filter instance
    kf = KalmanFilterWithControl(x0, P0, Q, R, dt)

    # Example noisy position measurements
    true_position = 0
    true_velocity = 1
    measurements = []
    estimated_positions = []
    estimated_velocities = []

    for _ in range(num_steps):
        true_position += true_velocity * dt + 0.5 * u[0] * dt**2  # Simulate motion
        true_velocity += u[0] * dt  # Update velocity
        
        measurement = true_position + np.random.normal(0, np.sqrt(R[0, 0]))  # Noisy measurement
        measurements.append(measurement)

        kf.predict(u)  # Predict step with control input
        kf.update(np.array([measurement]))  # Update step with measurement

        estimated_positions.append(kf.x[0])
        estimated_velocities.append(kf.x[1])

    # Print final estimated position and velocity
    print(f"Final Estimated Position: {kf.x[0]:.3f}")
    print(f"Final Estimated Velocity: {kf.x[1]:.3f}")
