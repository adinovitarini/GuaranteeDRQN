function yHat = InferenceConsig(model,sys,N,u,x)
%% This code is used for generate the sequence the estimated state that
% obtain from Luenberger and LQR 
A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;
Ry = model.Ry;
Ru = model.Ru;
% x = 0.1*ones(size(A,1),1);
% L = place(A',C',des_poles);
% [K,~,~] = lqr(sys,Ry,Ru);
for i = 1:N
   x(:,i+1) = A*x(:,i)+B*u(:,i);
   y(:,i) = C*x(:,i); 
   reward(:,i) = y(:,i)'*y(:,i)*Ry+u(:,i)'*u(:,i)*Ru;
end
yHat.x = x;
yHat.y = y;
yHat.u = u;
yHat.reward = reward;
end