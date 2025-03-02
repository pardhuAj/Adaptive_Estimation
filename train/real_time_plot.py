import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

class RealTimePlot:
    def __init__(self, env):
        self.env = env  # The environment instance
        self.fig, self.ax = plt.subplots(2, 1, figsize=(8, 6))
        
        # Initialize empty lists for plotting
        self.q_values = []
        self.r_values = []
        self.gt_q_values = []
        self.gt_r_values = []
        self.time_steps = []
        self.step_counter = 0
        self.window_size = 100  # Define moving window size
        
        # Create line objects
        self.q_line, = self.ax[0].plot([], [], 'r-', label='Q')
        self.gt_q_line, = self.ax[0].plot([], [], 'r--', label='GT_Q')
        self.r_line, = self.ax[1].plot([], [], 'b-', label='R')
        self.gt_r_line, = self.ax[1].plot([], [], 'b--', label='GT_R')
        
        # Set labels and legends
        self.ax[0].set_ylabel("Q Values")
        self.ax[1].set_ylabel("R Values")
        self.ax[1].set_xlabel("Time Step")
        
        self.ax[0].legend()
        self.ax[1].legend()
        
        # Set axis limits (dynamic)
        self.ax[0].set_xlim(0, self.window_size)
        self.ax[1].set_xlim(0, self.window_size)
        self.ax[0].set_ylim(0, 1)
        self.ax[1].set_ylim(0, 1)
        
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, interval=500, blit=False)
        
    def update_plot(self, frame):
        # Update values
        self.q_values.append(self.env.Q)
        self.r_values.append(self.env.R)
        self.gt_q_values.append(self.env.GT_Q)
        self.gt_r_values.append(self.env.GT_R)
        self.time_steps.append(self.step_counter)
        self.step_counter += 1
        
        # Trim lists to maintain moving window
        if len(self.time_steps) > self.window_size:
            self.q_values.pop(0)
            self.r_values.pop(0)
            self.gt_q_values.pop(0)
            self.gt_r_values.pop(0)
            self.time_steps.pop(0)
        
        # Update data
        self.q_line.set_data(self.time_steps, self.q_values)
        self.gt_q_line.set_data(self.time_steps, self.gt_q_values)
        self.r_line.set_data(self.time_steps, self.r_values)
        self.gt_r_line.set_data(self.time_steps, self.gt_r_values)
        
        # Dynamically adjust x-limits for moving window effect
        self.ax[0].set_xlim(max(0, self.step_counter - self.window_size), self.step_counter)
        self.ax[1].set_xlim(max(0, self.step_counter - self.window_size), self.step_counter)
        
        # Dynamically adjust y-limits
        self.ax[0].set_ylim(min(self.q_values + self.gt_q_values) - 0.1, max(self.q_values + self.gt_q_values) + 0.1)
        self.ax[1].set_ylim(min(self.r_values + self.gt_r_values) - 0.1, max(self.r_values + self.gt_r_values) + 0.1)
        
        return self.q_line, self.gt_q_line, self.r_line, self.gt_r_line
    
    def update(self):
        self.update_plot(None)
        plt.pause(0.001)  # Pause for real-time updating
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    from adaptive_rl_env import AdaptiveRLEnv  # Import your environment class
    
    env = AdaptiveRLEnv(seed=42)  # Initialize the environment
    plotter = RealTimePlot(env)  # Create a real-time plot instance
    
    # Assign plotter to the environment for real-time updates
    env.plotter = plotter
    
    plotter.show()  # Show the animated plot