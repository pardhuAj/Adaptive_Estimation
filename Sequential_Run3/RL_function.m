function [reward] = RL_function(Q, R,y)
        % Kalman gain calculation

        K = P_pred*Hx'*pinv(Hx*P_pred*Hx'+R); % Kalman gain eqaution
    
    % Update equations
    % State 
        xest_plus = xest_minus + K*(y - Hx*xest_pred);
        % Error covariance update
        P_Upd = (eye(length(xest_plus))-K*Hx)*P_minus;

    % Propagation
    % State propagation
        xest_minus = F*xest_plus+ H_p*pe0;
     % covariance propagation
        P_minus = F*P_Upd*F'+Q;
     reward = K*(y - Hx*xest_pred);
end