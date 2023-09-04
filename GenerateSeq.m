function dataset = GenerateSeq(sys,N,Ry,Ru)
%% This code is used for generate the sequence the estimated state that
% obtain from Luenberger and LQR 
A = sys.A;
B = sys.B;
C = sys.C;
D = sys.D;
x = 0.1*ones(size(A,1),1);
% L = place(A',C',des_poles);
[K,~,~] = lqr(sys,Ry,Ru);
for i = 1:N
   x(:,i+1) = (A-B*K)*x(:,i);
   y(:,i) = C*x(:,i); 
   u(:,i) = K*x(:,i);
   reward(:,i) = y(:,i)'*y(:,i)*Ry+u(:,i)'*u(:,i)*Ru;
end
dataset.x = x;
dataset.y = y;
dataset.u = u;
dataset.reward = reward;
end