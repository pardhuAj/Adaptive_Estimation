import numpy as np
from nonlintire import nonlintire
import matplotlib.pyplot as plt

class VehicleModel:
    def __init__(self, velocity):
        # Vehicle Parameters 0f BMW 535 Xi
        self.m = 1740.2  # kg
        self.E = 1.555  # Track width in m
        self.l = 2.885  # Wheelbase
        self.lf = 0.5025 * self.l  # CG distance to front in m
        self.lr = 0.4975 * self.l # CG distance to rear in m
        self.hf = 0.33 * 0.3048  # Roll center height front in m
        self.hr = 0.5 * 0.3048  # Roll center height rear in m
        self.hcg = 0.778  # Height CG
        self.g = 9.80665  # Gravitational acceleration
        self.kf = 0.66  # Roll stiffness front in % (78.87 N/mm)
        self.kr = 0.34  # Roll stiffness rear in % (41.13 N/mm)
        self.KR = 50000  # Total roll stiffness in Nm/rad
        self.CR = 3500  # Total roll damping in Nms/rad
        self.Iz = 3326  # Yaw plane moment of inertia
        self.Ix = 679  # Roll plane moment of inertia
        
        self.SR = 15; # Steering ratio
        self.x =[] # Initial Conditions for States
        # Calculating static wheel loads and CG-roll center distance
        self.Fzf = self.m * self.g * self.lr / self.l
        self.Fzf0 = self.Fzf / 2
        self.Fzr = self.m * self.g * self.lf / self.l
        self.Fzr0 = self.Fzr / 2
        self.hcr = self.hcg - (self.hf + (self.lf * (self.hr - self.hf) / self.l))
        self.cf = 2*802 * 180 / np.pi  # N/rad
        self.cr = 2*785 * 180 / np.pi  # N/rad
        self.vel = velocity  # Initial velocity in m/s
    def fishhook(self,time, dt):
        index1, index2 = int(20 / dt) + 1, int(22 / dt) + 1
        delta_sw = np.zeros(len(time))
        delta_sw[:index1] = 13.5 * time[:index1] # 13.5 deg/s
        delta_sw[index1:index2] = 270 # 270 deg
        delta_sw[index2:] = 270 - (67.5 * (time[index2:] - 22))
        return delta_sw, (delta_sw / self.SR) * (np.pi / 180)  # hand wheel angle in deg and road wheel in rad
    
    def Constant_Steering(self, time, dt):
        delta_sw = np.zeros(len(time))
        index1, index2, index3 = int(7 / dt) + 1, int(67 / dt) + 1, int(125 / dt) + 1
        delta_sw[index1:index2] = 45 
        deltavec = delta_sw * np.pi / 180
        return delta_sw, deltavec / self.SR # hand wheel angle in deg and road wheel in rad

    def slalom_steering(self,time, amplitude = 150, freq =0.3):
        delta_sw = np.zeros(len(time))
        delta_sw = amplitude * np.sin(2 * np.pi * freq * time)  # Sinusoidal input
        deltavec = np.deg2rad(delta_sw)  # Convert to radians
        return delta_sw, deltavec/self.SR # Road wheel angle

    def bicycle_model_nonlinear(self, x, delta,v):

        # unpacking the states
        beta, r, xcg, ycg, psi = x
        # Vehicle slip angles
        alphaf = delta - beta - r * self.lf / v
        alphar = -beta + r * self.lr / v

        # Wheel velocities
        vf = v * np.cos(beta) - r * self.lf * np.sin(beta)  # Front wheel velocity
        vr = v * np.cos(beta) + r * self.lr * np.sin(beta)  
        # Tire forces
        Fyf = -nonlintire(alphaf, self.Fzf,self.vel)  # front tire force
        Fyr = -nonlintire(alphar, self.Fzr,self.vel)  # Rear tire force
        betadot = -r + (Fyf  + Fyr) / self.m / self.vel
        rdot = (self.lf * Fyf - self.lr * Fyr) / self.Iz

        # Vehicle motion equations
        xcgdot = v * np.cos(beta + psi)
        ycgdot = v * np.sin(beta + psi)
        psidot = r

        return np.array([betadot, rdot, xcgdot, ycgdot, psidot])

    def run_simulation(self,id):
        if id ==1:
            T, delta_t = 26, 0.001
            time = np.linspace(0, T, int(T / delta_t) + 1)
            delta_sw, deltavec = self.fishhook(time, delta_t) # Steering input
            EntrySpeed = self.vel # Initial speed in m/s
        elif id ==2:
            T, delta_t = 75, 0.001
            time = np.linspace(0, T, int(T / delta_t) + 1)
            delta_sw, deltavec = self.Constant_Steering(time, delta_t) # Steering input
            EntrySpeed = self.vel  # Initial speed in m/s
        else:
            T, delta_t = 20, 0.001
            time = np.linspace(0, T, int(T / delta_t) + 1)
            delta_sw, deltavec = self.slalom_steering(time) # Steering input in rad
            EntrySpeed = self.vel 

        x = np.zeros((5, len(time))) # Initialization of state vector
        x[:, 0] = [ 0, 0, 0, 0, 0] # Initial states
        ay = np.zeros(len(time)+1) # Initialization of lateral acceleration
        for i in range(len(time) - 1):
            v = EntrySpeed; 
            xdot = self.bicycle_model_nonlinear(x[:, i], deltavec[i],v)
            x[:, i + 1] = x[:, i] + delta_t * xdot
            ay[i] = v*(xdot[0] + x[1,i]) # Lateral acceleration V*(r + beta)
        alphaf = deltavec - x[1, :] * self.lf / v - x[0, :]
        alphar = x[1, :] * self.lr /v - x[0, :]
        
        self.plot_results(time, x, deltavec,alphar,alphaf,v)

        return time, x, deltavec,ay,alphar,alphaf

    def plot_results(self,time, x,deltavec,v,alphar, alphaf):
        # Global velocity vector, vehicle heading direction and global position of CG
        
        fig, axs = plt.subplots()
        
        N = 1000; # Number of points to plot
        Vxquiv = v*np.cos(x[0,:]+x[1,:]) # Global x velocity vector starting point
        Vyquiv = v*np.sin(x[0,:]+x[1,:]) # Global y velocity vector staring point
        '''Vxquiv = Vxquiv[::int(len(time)/N)] # Downsample
        Vyquiv = Vyquiv[::int(len(time)/N)] # Downsample  

        Xquiv = x[2,::int(len(time)/N)] # Downsample CG x-coordinate
        Yquiv = x[3,::int(len(time)/N)] # Downsample CG y-coordinate
        
        axs.quiver(-x[3,::N],x[2,::N],-Vyquiv[::N],Vxquiv[::N],scale=10,color='black',label = 'velocity') # Plot velocity vector
        normalize = np.sqrt(np.sin(x[4, ::N])**2 + np.cos(x[4, ::N])**2)
        axs.quiver(-x[3,::N], x[2,::N], -np.sin(x[4, ::N])/normalize, np.cos(x[4, ::N])/normalize, scale=10,color='gray',label = 'Heading')''' # Plot heading direction'''
        axs.plot(-x[3, :], x[2, :], color = 'blue',label='Vehicle Path') # Plot vehicle co-ordinates
        axs.set_xlabel('y-direction')
        axs.set_ylabel('x-direction')
        axs.legend()
        axs.grid()
        axs.axis('equal')
        axs.set_title('Global Position of CG')
        plt.show()
        # Plot States
        fig, axs = plt.subplots(5, 1, figsize=(12, 15))
        titles = ['Vehicle slip angle', 'Yaw rate', 
                'x Global Coordinate', 'y Global Coordinate', 'Yaw Angle']
        
        for i in range(5):
            axs[i].plot(time, x[i , :])
            axs[i].set_title(titles[i])
            axs[i].set(xlabel='Time (s)')
            axs[i].grid()
        
        plt.tight_layout()
        plt.show()
        # Plot Steering Input
        ''''
        plt.figure()
        plt.plot(time, deltavec * self.SR*180/np.pi, label='Steering Wheel Angle (degrees)')
        plt.plot(time, deltavec * 180 / np.pi, label='Road wheel in Degrees')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid()
        plt.title('Steering Input')
        plt.show()
        # Plot Slip angles
        plt.figure()
        plt.plot(time, alphaf*180/np.pi, label='Front slip angle in degrees')
        plt.plot(time, alphar*180/np.pi, label='Rear slip angle in degrees')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid()
        plt.title('Tire Slip angles')
        plt.show()
        '''


if __name__ == "__main__":
    model = VehicleModel(50/2)
    #model.run_simulation(1) # Fishhook maneuver
    model.run_simulation(1) # Constant steering
    #model.run_simulation(3) # Slalom maneuver
