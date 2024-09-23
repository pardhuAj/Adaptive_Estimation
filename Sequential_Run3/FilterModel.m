
function send = FilterModel(Q,R,xest_upd_pr,P_Upd_pr,y,n_steps)
 
    A = [ 0.75 -1.74  -0.3 0 -0.15;
    0.09 0.91 -0.0015 0 -0.008;
    0 0 0.95 0 0;
    0 0 0 0.55 0;
    0 0 0 0 0.905];

    B = [0 0 0;
    0 0 0;
    24.64 0 0;
    0 0.835 0;
    0 0 1.83];
    
    H = [1 0 0 0 1;
    0 1 0 1 0];
    
    % Propagation
    xest_pred = A*xest_upd_pr;
    P_pred = A*P_Upd_pr*A'+B*Q*B';
    K = P_pred*H'*pinv(H*P_pred*H'+R); % Kalman gain eqaution
    % Update equations
    xest_upd = xest_pred + K*(y(:,n_steps) - H*xest_pred);
    delta =   K*(y(:,n_steps) - H*xest_pred);
    P_Upd = (eye(length(xest_upd))-K*H)*P_pred;
    

    %send = [delta,xest_upd,P_Upd];
    send.delta = delta;
    send.xest_upd = xest_upd;
    send.P_Upd = P_Upd;

end