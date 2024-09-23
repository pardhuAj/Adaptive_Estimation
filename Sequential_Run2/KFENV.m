
classdef KFENV < rl.env.MATLABEnvironment
    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Initializations for environments/ states
        y = [];
        xest_upd_pr = [1;1];
        P_Upd_pr = diag([1,1]);
        Reward = [];
        delta = [];
        steps = 1;
        nsteps = [];
        Q = [];
        R = [];   
    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = KFENV(inputs)
            % Initialize Observation settings
            ObservationInfo = rlNumericSpec([3,1]);
            ObservationInfo.Name = 'Measurement (each timestep) Delta (state1) Delta(state2)';
            ObservationInfo.Description = 'y delta(x1) delta(x2)';
            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec([2,1],'LowerLimit',[0.0001;0.0001] ,'UpperLimit',[1;1]); % N
            ActionInfo.Name = 'Covariances';
            ActionInfo.Description = 'Q R';
            
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            %this.train = inputs.train;
            this.y = inputs.y;
            this.nsteps = inputs.nsteps;
            %this.Ts = inputs.Ts;
            %this.Tf = this.nsteps*this.Ts;

        end
        %End of initialization function
        
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            
            LoggedSignals = [];
            %fprintf('Here')     
            covariance = double(Action);
            this.Q = covariance(1);
            this.R = covariance(2);

            %%%%%% ADD STEP Function here
            send = FilterModel(this.Q,this.R,this.xest_upd_pr,this.P_Upd_pr,this.y,this.steps);
            this.xest_upd_pr = send.xest_upd;
            this.P_Upd_pr = send.P_Upd;
            this.delta = send.delta;
      
            Observation = [this.y(this.steps);send.delta(1);send.delta(2)]; % 3 by 1 vector

            % Get reward
            Reward = getReward(this);
            
            %Print Values
            %fprintf('T = [%0.2f %0.2f %0.2f %0.2f]\t X_e = [%0.2f %0.2f]\t distance_error = %0.2f\t X_init = [%0.2f %0.2f]\t X_des = [%0.2f %0.2f]\t Rewards = [%0.2f %0.2f %0.2f %0.2f] \n' ...
            %    ,joint_state(1),joint_state(2),joint_state(3),joint_state(4),this.X(1),this.X(2),dist_error,this.X_init(1),this.X_init(2),this.X_des(1),this.X_des(2),this.R0,this.R1,this.R2,this.R3)
             
            % Termination criterion if any

            if this.steps == this.nsteps
                IsDone = true;
            else
                IsDone = false;
            end

            this.steps = this.steps + 1;
                
        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this)

                        
            Observation = [-0.0669112995372736 ;1;1]; % 3 by 1 vector

            InitialObservation = Observation;
            
            this.Reward = 0;
            this.steps = 1;
            %this.episodes = this.episodes +1;
                
        end
        %End of Reset function
        
    end
    %End of Necessary Methods
    %% Optional Methods (set methods' attributes accordingly)
    methods               
        % Reward function
        function Reward = getReward(this)
            this.Reward = -1*norm(this.delta);
            Reward = this.Reward;
            
        end   
    end
    
    methods (Access = protected)
        end
    end

