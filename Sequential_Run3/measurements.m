clc;clear;
dt = 0.1;
phi = [ 0.75 -1.74  -0.3 0 -0.15;
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

Q = eye(3);
R = eye(2);
[Pss,~,~] = idare(phi',H',B*Q*B',R,[],[]);
Wss = (Pss*H')/(H*Pss*H'+R); % Initial kalman gain
% Creating a random sequence
rng(0,'twister');

time = 0:dt:100;

wk = sqrt(Q)*randn(3,length(time));
vk = sqrt(R)*randn(2,length(time));

% True model

x = zeros(size(phi,1),length(time));
x(:,1) = ones(size(phi,1),1);
for i = 1:length(time)-1
    x(:,i+1) = phi*x(:,i) + B*wk(:,i);
end
y= H*x + vk;
save('y.mat','y')
clear