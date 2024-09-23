
classdef KFENV < rl.env.MATLABEnvironment
    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Initializations for environments/ states
        M = 100;
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
            ObservationInfo = rlNumericSpec([100,1]);
            ObservationInfo.Name = 'Sample covaraince arrays';
            ObservationInfo.Description = 'C_estimate';
            
            % Initialize Action settings   
            ActionInfo = rlNumericSpec([2,1],'LowerLimit',[0.0001;0.0001] ,'UpperLimit',[1;1]); % N
            ActionInfo.Name = 'Covariances';
            ActionInfo.Description = 'Q R';
            
            
            % The following line implements built-in functions of RL env
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            %this.train = inputs.train;

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
            send = FilterModel(this.Q,this.R,this.M);
            this.delta = reshape(send.delta,[],1);
      
            Observation = this.delta; % 3 by 1 vector

            % Get reward
            Reward = getReward(this);
            
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

                        
            Observation = ones(this.M,1); % M by 1 vector

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
            kmax = max(this.delta);
            kmin = min(this.delta);
            s = (this.delta - kmin)/(kmax-kmin);
            this.Reward = (1-(norm(s)/10))^2;
            Reward = this.Reward;   
        end   
    end
    
    methods (Access = protected)
        end
end

