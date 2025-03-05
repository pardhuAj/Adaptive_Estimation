import numpy as np
import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/")
from nonlintire import nonlintire
import matplotlib.pyplot as plt

class fourwheel_model:
    def __init__(self,velocity,randM,randIz):
        # Vehicle Parameters 0f BMW 535 Xi
        self.m = 1740.2 + randM # kg
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
        self.Iz = 3326 + randIz  # Yaw plane moment of inertia
        self.Ix = 679  # Roll plane moment of inertia
        self.Velocity = velocity
        
        self.SR = 15; # Steering ratio
        self.x =[] # Initial Conditions for States
        # Calculating static wheel loads and CG-roll center distance
        self.Fzf = self.m * self.g * self.lr / self.l
        self.Fzf0 = self.Fzf / 2
        self.Fzr = self.m * self.g * self.lf / self.l
        self.Fzr0 = self.Fzr / 2
        self.hcr = self.hcg - (self.hf + (self.lf * (self.hr - self.hf) / self.l))
    
    def fishhook(self,time, dt):
        index1, index2 = int(20 / dt) + 1, int(22 / dt) + 1
        delta_sw = np.zeros(len(time))
        delta_sw[:index1] = 13.5 * time[:index1] # 13.5 deg/s
        delta_sw[index1:index2] = 270 # 270 deg
        delta_sw[index2:] = 270 - (67.5 * (time[index2:] - 22))
        return delta_sw, (delta_sw / self.SR) * (np.pi / 180)  
    
    def Constant_Steering(self, time, dt):
        delta_sw = np.zeros(len(time))
        index1, index2, index3 = int(7 / dt) + 1, int(67 / dt) + 1, int(125 / dt) + 1
        delta_sw[index1:index2] = 45 / self.SR
        deltavec = delta_sw * np.pi / 180
        return delta_sw, deltavec

    def slalom_steering(self,time, amplitude = 150, freq =0.3):
        delta_sw = np.zeros(len(time))
        delta_sw = amplitude * np.sin(2 * np.pi * freq * time)  # Sinusoidal input
        deltavec = np.deg2rad(delta_sw)  # Convert to radians
        return delta_sw/self.SR, deltavec/self.SR # Road wheel angle
    
    def simulation(self,x, delta, ay):

        vxmin = 0.01 # Minimum velocity in longitiudinal direction 

        # NO driving force at wheels
        Fx11 = 0
        Fx12 = 0
        Fx21 = 0
        Fx22 = 0

        # unpack the states
        r = x[0] # yaw rate
        beta = x[1] # Vehicle side slip angle
        Vg = x[2] # Velocity of CG
        vx = max(x[3], vxmin) # Longitudinal velocity
        vy = x[4] # Lateral velocity
        xglobal = x[5] # x-coordinate of CG
        yglobal = x[6] # y-coordinate of CG
        psi = x[7] # yaw angle
        phi1 = x[8] # roll angle
        phi2 = x[9] # roll rate

        # Vehicle wheel velocities
        vwx11 = (vx-r*self.E/2) * np.cos(delta) + (vy + r*self.lf) * np.sin(delta)
        vwx12 = (vx+r*self.E/2) * np.cos(delta) + (vy + r*self.lf) * np.sin(delta)
        vwx21 = (vx-r*self.E/2)
        vwx22 = (vx+r*self.E/2)
        # Quasi static tire slip angles
        alpha11 = delta - np.arctan((vy+self.lf*r)/(vx-(self.E*r/2)))
        alpha12 = delta - np.arctan((vy+self.lf*r)/(vx+(self.E*r/2)))
        alpha21 = -np.arctan((vy-self.lr*r)/(vx-(self.E*r/2)))
        alpha22 = -np.arctan((vy-self.lr*r)/(vx+(self.E*r/2)))

        # Tire Loads
        ef = self.E
        er = self.E
        # Setting 1: No load transfer -> activate this part

        # delta_Fz1 = 0
        # delta_Fz2 = 0

        # Setting 2: Load transfer due to steady-state roll -> activate this part

        # phi_ss = -self.m*ay*self.hcr/(self.KR - (self.m*self.hcr*self.g))
        # phisave(j,1) = phi_ss
        # delta_Fz1 = -((self.kf*self.KR)/ef)*phi_ss + (self.m*ay*self.lr*self.hf)/(self.l*self.ef)
        # delta_Fz2 = -((self.kr*self.KR)/self.er)*phi_ss + (self.m*ay*self.lf*self.hr)/(self.l*self.er)

        # Setting 3: Load transfer due to dynamic roll -> activate this part


        delta_Fz1 = -(self.kf*self.KR/ef)*phi1 + (self.m*ay*self.lr*self.hf)/(self.l*ef)
        delta_Fz2 = -(self.kr*self.KR/er)*phi1 + (self.m*ay*self.lf*self.hr)/(self.l*er)

        # Tire load updates

        Fz11 = self.Fzf0 - delta_Fz1
        Fz12 = self.Fzf0 + delta_Fz1
        Fz21 = self.Fzr0 - delta_Fz2
        Fz22 = self.Fzr0 + delta_Fz2
        '''
        Fz11s(j,1) = Fz11;
        Fz12s(j,1) = Fz12;
        Fz21s(j,1) = Fz21;
        Fz22s(j,1) = Fz22;
        '''
        # New tire lateral forces

        Fy11 = -nonlintire(alpha11,Fz11,vwx11);
        Fy12 = -nonlintire(alpha12,Fz12,vwx12);
        Fy21 = -nonlintire(alpha21,Fz21,vwx21);
        Fy22 = -nonlintire(alpha22,Fz22,vwx22);
        '''
        # Longitudinal forces
        Fx21 = Fy11*sin(delta) + Fy12*sin(delta)
        Fx22 = Fy11*sin(delta) + Fy12*sin(delta)
        '''
        # State updates

        rdot = (1/self.Iz)* (self.lf*(Fy11*np.cos(delta) + Fy12*np.cos(delta) + Fx11*np.sin(delta) + Fx12*np.sin(delta))- self.lr*(Fy21 + Fy22)+ (self.E/2)*(Fy11*np.sin(delta) - Fy12*np.sin(delta) + Fx12*np.cos(delta) - Fx11*np.cos(delta) + Fx22 - Fx21)) # yaw rate

        betadot = -r + (1/(self.m*Vg)) * (-(Fx11+Fx12)*np.sin(beta-delta) + Fy11*np.cos(beta-delta) + Fy12*np.cos(beta-delta) + (Fy21+Fy22)*np.cos(beta) - (Fx21+Fx22)*np.sin(beta)) # body slip angle rate
        Vgdot = (1/self.m)*((Fx11+Fx12)*np.cos(beta-delta) + Fy11*np.sin(beta-delta)+ Fy12*np.sin(beta-delta)+(Fx21+Fx22)*np.cos(beta) +(Fy21+Fy22)*np.sin(beta)) # body acceleration along Velocity of CG

        ay = (1/self.m)*(Fy11*np.cos(delta) + Fy12*np.cos(delta) + (Fy21+Fy22) + Fx11*np.sin(delta) + Fx12*np.sin(delta)) # body acceleration in y-direction
        ax = (1/self.m)*((-Fy11*np.sin(delta) - Fy12*np.sin(delta) + Fx12*np.cos(delta) + Fx11*np.cos(delta) + Fx22 + Fx21)) # body acceleration in x-direction

        vxdot = ax + r*vy # Global x velocity expressed in body frame
        vydot = ay - r*vx # Global y velocity expressed in body frame

        xglobaldot = vx * np.cos(psi) - vy * np.sin(psi) # Global x velocity expressed in global frame
        yglobaldot = vx * np.sin(psi) + vy * np.cos(psi) #
        
        psidot = r # yaw angle rate
        phi1dot = phi2 # roll angle rate
        phi2dot = (1 / self.Ix) * (-self.m * ay * self.hcr + self.m * self.hcr * self.g * np.sin(phi1) - self.CR * phi2 - self.KR * phi1)
        
        return np.array([rdot, betadot, Vgdot, vxdot, vydot, xglobaldot, yglobaldot, psidot, phi1dot, phi2dot]), ay

    def run_simulation(self,id):
        if id ==1:
            T, delta_t = 25, 0.01
            time = np.linspace(0, T, int(T / delta_t) + 1)
            delta_sw, deltavec = self.fishhook(time, delta_t) # Steering input
            EntrySpeed = self.Velocity
        elif id ==2:
            T, delta_t = 75, 0.01
            time = np.linspace(0, T, int(T / delta_t) + 1)
            delta_sw, deltavec = self.Constant_Steering(time, delta_t) # Steering input
            EntrySpeed = self.Velocity
        else:
            T, delta_t = 20, 0.01
            time = np.linspace(0, T, int(T / delta_t) + 1)
            delta_sw, deltavec = self.slalom_steering(time) # Steering input in rad
            EntrySpeed = self.Velocity

        x = np.zeros((10, len(time))) # Initialization of state vector
        x[:, 0] = [0, 0, EntrySpeed, EntrySpeed, 0, 0, 0, 0, 0, 0] # Initial states
        ay = np.zeros((1,len(time)+1)) # Initialization of lateral acceleration
        for i in range(len(time) - 1):
            xdot , ay[0,i+1] = self.simulation(x[:, i], deltavec[i],ay[0,i])
            x[:, i + 1] = x[:, i] + delta_t * xdot
        
        #self.plot_results(time, x, deltavec)
            
        return {
            "time" : time,
            "x" : x,  # Added simulated measurements
            "deltavec" : deltavec,
            "ay" : ay,
              # Store for logging
        }
        #return time, x, deltavec,ay
    

    def plot_results(self,time, x,deltavec):
        # Global velocity vector, vehicle heading direction and global position of CG
        fig, axs = plt.subplots()
        N = 1000; # Number of points to plot
        x3quiv = x[3,:]*np.cos(x[7,:])-x[4,:]*np.sin(x[7,:]) # Global x velocity vector starting point
        x4quiv = x[3,:]*np.sin(x[7,:])+x[4,:]*np.cos(x[7,:]) # Global y velocity vector staring point
        x3quiv = x3quiv[::int(len(time)/N)] # Downsample
        x4quiv = x4quiv[::int(len(time)/N)] # Downsample  

        x5quiv = x[5,::int(len(time)/N)] # Downsample CG x-coordinate
        x6quiv = x[6,::int(len(time)/N)] # Downsample CG y-coordinate

        axs.quiver(-x6quiv,x5quiv,-x4quiv,x3quiv,scale=10,color='black',label = 'velocity') # Plot velocity vector
        normalize = np.sqrt(np.sin(x[7, ::N])**2 + np.cos(x[7, ::N])**2)
        axs.quiver(-x[6,::N], x[5,::N], -np.sin(x[7, ::N])/normalize, np.cos(x[7, ::N])/normalize, scale=10,color='gray',label = 'Heading') # Plot heading direction
        axs.plot(-x[6, :], x[5, :], color = 'blue',label='Vehicle Path') # Plot vehicle co-ordinates
        axs.set_xlabel('y-direction')
        axs.set_ylabel('x-direction')
        axs.legend()
        axs.grid()
        axs.axis('equal')
        axs.set_title('Global Position of CG')
        plt.show()
        # Plot States
        fig, axs = plt.subplots(5, 2, figsize=(12, 15))
        titles = ['Yaw Rate', 'Vehicle Slip', 'CG Velocity', 'Vx', 'Vy',
                'x Global Coordinate', 'y Global Coordinate', 'Yaw Angle',
                'Roll Angle', 'Roll Velocity']
        
        for i in range(5):
            for j in range(2):
                axs[i, j].plot(time, (x[i * 2 + j, :]))
                axs[i, j].set_title(titles[i * 2 + j])
                axs[i, j].set(xlabel='Time (s)')
                axs[i, j].grid()
        
        plt.tight_layout()
        plt.show()
        # Plot Steering Input
        plt.figure()
        plt.plot(time, deltavec * self.SR*180/np.pi, label='Steering Wheel Angle (degrees)')
        plt.plot(time, deltavec * 180 / np.pi, label='Road wheel in Degrees')
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid()
        plt.title('Steering Input')
        plt.show()

if __name__ == "__main__":
    randM = np.random.uniform(0,500)
    randIz = np.random.uniform(0,100)
    print(f"Rand M is:{randM}")
    print(f"Rand Iz is:{randIz}")
    model = fourwheel_model(randM,randIz)
    fishhook = model.run_simulation(1)
    #constant_steering = model.run_simulation(2) 
    #slalom = model.run_simulation(3)
