
function send = FilterModel(Q,R,xest_upd_pr,P_Upd_pr,y,n_steps)
 
    dt = 0.1; 
    A = [1 dt;0 1];
    H = [1 0];
    B = [0.5*dt^2; dt];
    % Propagation
    xest_pred = A*xest_upd_pr;
    P_pred = A*P_Upd_pr*A'+B*Q*B';
    K = P_pred*H'*pinv(H*P_pred*H'+R); % Kalman gain eqaution
    % Update equations
    xest_upd = xest_pred + K*(y(n_steps) - H*xest_pred);
    delta =   K*(y(n_steps) - H*xest_pred);
    P_Upd = (eye(length(xest_upd))-K*H)*P_pred;
    

    %send = [delta,xest_upd,P_Upd];
    send.delta = delta;
    send.xest_upd = xest_upd;
    send.P_Upd = P_Upd;

end