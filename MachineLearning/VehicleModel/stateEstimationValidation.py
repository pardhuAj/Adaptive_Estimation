import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import Kalman filter and vehicle dynamics model
import sys
sys.path.insert(0, "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/")
from VDkalman_filter_control import KalmanFilterWithControl
from VDlabelQR import LabelQR
from VDdata_gen_dataset import DataGenerator  # ✅ Importing DataGenerator directly

from VDFourwheelmodel_Plots import fourwheel_model
from VDBIcyclemodel_BMW import VehicleModel

# ------------------------------ NEURAL NETWORK MODEL -------------------------------- #
class KalmanNet(nn.Module):
    def __init__(self, input_size=203, hidden_size=128):  # Updated input size
        super(KalmanNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # Output Qa_hat, Qb_hat, R_hat

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.relu(x)  # Ensure Q_hat and R_hat are non-negative

# ------------------------------ VALIDATION FUNCTION -------------------------------- #
class ValidateModel:
    def __init__(self, model_path):
        #self.num_samples = num_samples
        self.model = KalmanNet().cuda()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # ✅ Use DataGenerator to create new samples
        self.data_generator = DataGenerator(num_samples=1)

    def run(self, output_csv="validation_results.csv"):
        Qa_hat_values, Qb_hat_values, R_hat_values = [], [], []
        Qa_true_values, Qb_true_values, R_true_values = [], [], []
        beta_true_all, yaw_true_all = [], []
        Xcg_true_all, Ycg_true_all = [], []  # ✅ New tracking
        beta_hat_all, yaw_hat_all = [], []
        P_updated_all = []

        #print(f"Starting validation with {self.num_samples} generated samples...")

        #for _ in tqdm(range(self.num_samples), desc="Running Validation"):

        # Generate new sample using DataGenerator
        #sample = self.data_generator.generate_single_sample()
        #print(sample)

        #This time, not gerating a random sample but actually running 1) Initialization with first 100 steps and then beyound that reccursively

        tmz =[]
        Qa_true_values = []
        Qb_true_values = []
        R_true_values = []

        residual_measurements = []
        x_hat_states = []
        P_updated = []

        LbQR = LabelQR(k=1)
        values = LbQR.getQ()
        Q_true = values["Q"] #Label Q
        #print(Q_true)
        R_true = np.random.uniform(0, 0.1) #Label R

        #k = values["k"] #from where to begin the traning sample
        Qa_true_values = [Q_true[0,1]*100]
        Qb_true_values = [Q_true[1,1]*100]
        R_true_values = [R_true*100]

        #We generate true measurement by adding Q and R to the non-varying bicyle model
        true_measurements = values["TrueMeasureYawRate"] + np.sqrt(Q_true[1,1])*np.random.normal(0,1,size=len(values["TrueMeasureYawRate"])) + np.sqrt(R_true)*np.random.normal(0,1,size=len(values["TrueMeasureYawRate"]))
        print(f"lenght of true_measurements{true_measurements.shape}")
        tmz.append(true_measurements)
        print(f"lenght of tmz{tmz}")

        #Call outside loop for ground truth
        
        plant_yawrate = values["TrueYawRate"] 
        plant_beta = values["TrueBeta"] 
        plant_Xcg = values["TrueXcg"] 
        plant_Ycg = values["TrueYcg"]

        Qa_dummy = np.random.uniform(0,0.1) 
        Qb_dummy = np.random.uniform(0,0.1)
        R_dummy = np.random.uniform(0,0.1)

        Qa_hat_values = [Qa_dummy]*100
        Qb_hat_values = [Qb_dummy]*100
        R_hat_values = [R_dummy]*100


        # Initialize Kalman filter for ground truth measurements
        kf_true = KalmanFilterWithControl(
            x0=np.array([0, 0]), 
            P0=np.array([[1, 0], [0, 1]]), 
            Q=np.diag([Qa_hat_values[-1],Qb_hat_values[-1]]), 
            R=R_dummy, 
            dt=0.01
        )


        for _ in range(100):
            kf_true.predict()
            kf_true.update(true_measurements[_])
            rm =  kf_true.y #get innovation sequence
            residual_measurements.append(rm.item())

            x_hat_states.append(kf_true.x.copy())  # Store adaptive estimates
            p_1 = np.sqrt(kf_true.P[0, 0])  # Uncertainty in beta
            p_2 = np.sqrt(kf_true.P[1, 1])  # Uncertainty in yaw
            P_updated.append([p_1, p_2])



        for k in range(101,1500):

            LbQR = LabelQR(k)
            values = LbQR.getQ()
            Q_true = values["Q"] #Label Q
            #print(Q_true)
            R_true = np.random.uniform(0, 0.1) #Label R

            #k = values["k"] #from where to begin the traning sample
            Qa_true_values.append(Q_true[0,1])
            Qb_true_values.append(Q_true[1,1])
            R_true_values.append(R_true)

            #We generate true measurement by adding Q and R to the non-varying bicyle model
            true_measurements = values["TrueMeasureYawRate"] + np.sqrt(Q_true[1,1])*np.random.normal(0,1,size=len(values["TrueMeasureYawRate"])) + np.sqrt(R_true)*np.random.normal(0,1,size=len(values["TrueMeasureYawRate"]))
            print(f"Inside the loop lenght of true_measurements{true_measurements.shape}")
            tmz.append(true_measurements[0])
            print(f"tmz inside loop{tmz}")

            #Call outside loop for ground truth
            
            #plant_yawrate = values["TrueYawRate"] 
            #plant_beta = values["TrueBeta"] 
            #plant_Xcg = values["TrueXcg"] 
            #plant_Ycg = values["TrueYcg"]
            

            ############ Generate the set of first 100 measuremetns for initializatio

                
                

                ### save the states as well so that the new loop can be be initialized with those states


            ################## This generates the set of first 100 measurements
            '''
            #Run first prediction on these 100 measurements
                
            # Convert NumPy lists to NumPy arrays before converting to tensor
            input_measurements = torch.tensor(true_measurements[0:100], dtype=torch.float32).squeeze()
            #print(true_measurements.shape)
            residual_measurements = torch.tensor(residual_measurements[-100:], dtype=torch.float32).squeeze()
            #print(residual_measurements.shape)

            dummyQa = torch.tensor(Qa_dummy, dtype=torch.float32).unsqueeze(0)
            dummyQb = torch.tensor(Qb_dummy, dtype=torch.float32).unsqueeze(0)
            dummyR = torch.tensor(R_dummy, dtype=torch.float32).unsqueeze(0)


            # Concatenate inputs
            inputs = torch.cat([input_measurements, residual_measurements, dummyQa, dummyQb, dummyR]).cuda()


            predictions = self.model(inputs)  # Shape: [batch_size, 3]

            # Extract True Measurements from Inputs (Shape: [batch_size, 100])
            #true_measurementsZ = np.array(sample["true_measurements"][:100])

            # Extract Qa_hat, Qb_hat, R_hat for Each Batch (Shape: [batch_size])
            predictions = predictions.cpu().detach().numpy()  # Convert to NumPy
            Qa_hat, Qb_hat, R_hat = predictions[0], predictions[1], predictions[2]

            #Qa_hat_values.append(Qa_hat)
            #Qb_hat_values.append(Qb_hat)
            #R_hat_values.append(R_hat)
            '''

            ######## Now run the reccursive state measurement with model prediction
                
            #for i in range(100,len(plant_yawrate)):

            # Convert NumPy lists to NumPy arrays before converting to tensor
            input_measurements = torch.tensor(np.asanyarray(tmz[-100:]), dtype=torch.float32).squeeze()
            #print(true_measurements.shape)
            innovation_seq = torch.tensor(residual_measurements[-100:], dtype=torch.float32).squeeze()
            #print(residual_measurements.shape)

            dummyQa = torch.tensor(Qa_hat_values[-1], dtype=torch.float32).unsqueeze(0)
            dummyQb = torch.tensor(Qb_hat_values[-1], dtype=torch.float32).unsqueeze(0)
            dummyR = torch.tensor(R_hat_values[-1], dtype=torch.float32).unsqueeze(0)

            # Concatenate inputs
            inputs = torch.cat([input_measurements, innovation_seq, dummyQa, dummyQb, dummyR]).cuda()

            predictions = self.model(inputs)  # Shape: [batch_size, 3]

            # Extract True Measurements from Inputs (Shape: [batch_size, 100])
            #true_measurementsZ = np.array(sample["true_measurements"][:100])

            # Extract Qa_hat, Qb_hat, R_hat for Each Batch (Shape: [batch_size])
            predictions = predictions.cpu().detach().numpy()  # Convert to NumPy
            Qa_hat, Qb_hat, R_hat = predictions[0], predictions[1], predictions[2]

            Qa_hat_values.append(Qa_hat)
            Qb_hat_values.append(Qb_hat)
            R_hat_values.append(R_hat)

            # now you keep on running a single step kalman filter 
            kf_adaptive = KalmanFilterWithControl(
                x0= np.array(x_hat_states[-1]), 
                P0= np.array([[P_updated[-1][0], 0], [0, P_updated[-1][1]]]), 
                Q = [[Qa_hat.item(),0],[0,Qb_hat.item()]], 
                R = R_hat.item(), 
                dt=0.01
                )
            
            kf_adaptive.predict()
                #print(f"size of tm{np.shape(true_measurements)}")
            kf_adaptive.update(tmz[-1])
            rm =  kf_adaptive.y #get innovation sequence
            residual_measurements.append(rm.item())
            x_hat_states.append(kf_adaptive.x.copy())  # Store adaptive estimates
            p_1 = np.sqrt(kf_adaptive.P[0, 0])  # Uncertainty in beta
            p_2 = np.sqrt(kf_adaptive.P[1, 1])  # Uncertainty in yaw
            P_updated.append([p_1, p_2])

            print("One loop is over")


            

        
            '''
            modelBicyle = VehicleModel(25)
            BicyleManeuver = modelBicyle.run_simulation(1) 
            Bi_states = np.array(BicyleManeuver["x"])

            Bi_beta = Bi_states[0,:100] + np.random.normal(0,0.01*np.sqrt(sample["Qa_true"]))
            print(f"Bi_beta shape is : {np.shape(Bi_beta)}")
            Bi_yaw = Bi_states[1,:100] + np.random.normal(0,0.01*np.sqrt(sample["Qb_true"]))
            '''

        # Store logged data
        
        #Qa_true_values.append(sample["Qa_true"])
        #Qb_true_values.append(sample["Qb_true"])
        #R_true_values.append(sample["R_true"])

        # ✅ Log final values for comparison
        #beta_true_all.append(sample["plant_beta"])
        #yaw_true_all.append(sample["plant_yawrate"])
        #Xcg_true_all.append(sample["plant_Xcg"])
        #Ycg_true_all.append(sample["plant_Ycg"])
            
        #print(np.shape(x_hat_states))
        x_hat_states = np.array(x_hat_states)
        beta_hat_all = x_hat_states[:,0]
        yaw_hat_all = x_hat_states[:,1]
        P_updated_all = P_updated
        #print(np.shape(sample["plant_beta"][:100]))
        #print(np.shape(np.ones(100)*sample["Qa_true"]))
        #print(np.shape(beta_hat_all))

    #print(f"x_true shape: {len(yaw_true_all)}")
    #print(f"x_hat shape: {len(yaw_hat_all)}")
        
        output_len = len(plant_yawrate)

        # Convert data to DataFrame
        df = pd.DataFrame({
            "Qa_true": Qa_true_values,
            "Qb_true": Qb_true_values,
            "R_true": R_true_values,
            "Qa_hat": Qa_hat_values,
            "Qb_hat": Qb_hat_values,
            "R_hat": R_hat_values,
            "beta_true": plant_beta,
            "yawRate_true": plant_yawrate,
            "Xcg_true": plant_Xcg,
            "Ycg_true": plant_Ycg,
            "beta_hat": beta_hat_all,
            "yaw_hat": yaw_hat_all,
            "P1_updated": [x[0] for x in P_updated_all],
            "P2_updated": [x[1] for x in P_updated_all],
            #"biBeta" : Bi_beta,
            #"biYaw" : Bi_yaw
        })

        df.to_csv(output_csv, index=False)
        print(f"Validation results saved to {output_csv}")

# ------------------------------ RUN VALIDATION -------------------------------- #
if __name__ == "__main__":
    model_path = "/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/outputC/test_kalman_nn.pth"
    validator = ValidateModel(model_path)
    validator.run(output_csv="/home/asalvi/code_workspace/RL_AdpEst/MachineLearning/VehicleModel/outputC/test_validation_results.csv")
