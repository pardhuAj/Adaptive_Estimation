
function send = FilterModel(Q,R,M)
 
    dt = 0.1; 
    phi = [1 dt;0 1];
    B = [0.5*dt^2; dt];
    H = [1 0];
    time = 0:dt:100;
    Q_orig = 0.0025;
    R_orig = 0.01;
    wk = sqrt(Q_orig)*randn(1,length(time));
    vk = sqrt(R_orig)*randn(1,length(time));
    
    % True model
    
    x = zeros(size(phi,1),length(time));
    x(:,1) = ones(size(phi,1),1);
    for i = 1:length(time)-1
        x(:,i+1) = phi*x(:,i) + B*wk(i);
    end
    y= H*x + vk;

    % Initial values of W Q R and P -- step 1
    xest_pred = zeros(size(phi,1),length(time));
    xest_upd = zeros(size(phi,1),length(time));
    xest_init = 0.01*ones(size(phi,1),1);
    xest_upd(:,1) = xest_init;
    xest_pred(:,1) = xest_init;
  
    Q_0 = Q;
    R_0 = R;
    [P,~,~] = idare(phi',H',B*Q_0*B',R_0,[],[]);
    
    W = (P*H')/(H*P*H'+R_0); % Initial kalman gain
    N = size(y,2);
    W_0 = W;
    nu = zeros(size(y));
    mu = zeros(size(y));
    for i = 1:length(time)-1
        % State propagation equation
        xest_pred(:,i+1) = phi*xest_upd(:,i);
        % Innovation sequence with covariance S
        nu(:,i+1) = y(:,i+1) - H*xest_pred(:,i+1);
        % State update equation
        xest_upd(:,i+1) = xest_pred(:,i+1) + W_0*nu(:,i+1);
        % Post residual fit
        mu(:,i+1) =  y(:,i+1) - H*xest_upd(:,i+1);
    end
    %%% Generate covariance samples -- step 2

    C_est = zeros([size(vk,1),size(vk,1),M]);
    k = zeros([size(vk,1),size(vk,1),N-M]);
    
    for i = 1:M
        for j = 1:N-M
            k(:,:,j) = nu(:,j)*nu(:,j+i-1)';
        end
        C_est(:,:,i) = sum(k,3)/(N-M);
    end
    %send = [delta,xest_upd,P_Upd];
    send.delta = C_est;
end