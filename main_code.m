%% main code 
clear all;clc;
N = 10;
df = 1;
Ry = 1;
Ru = 1;
model.Ry = Ry;
model.Ru = Ru;
model.df = df;
model.N = N;
plant = sysmdl_cartpole(N,df);

sys = plant.sys;

dataset = GenerateSeq(sys,N,Ry,Ru);
x = dataset.x;
y = dataset.y;
u = dataset.u;
%%
A = sys.A;
B = sys.B;
x = x(:,1:N);
[P,K,G] = value_iteration(A,N,B,Ry,Ru);
uMRL = K(end,:)*x;
for i = 1:N 
    uu(:,i) = K(i,:)*x(:,i);
end
%%

dataset_ts = GenerateSeq(sys,N,Ry,Ru);
TRAIN.input = [dataset.u(:,1:N);dataset.y(:,1:N)];
TEST.input = [dataset_ts.u(:,1:N);dataset_ts.y(:,1:N)];
Q = q_func(model,dataset.reward);
Qts = q_func(model,dataset_ts.reward);
m = size(B,2);
for i = 1:m
    TRAIN.target(i,:) = Q;
    TEST.target(i,:) = Qts;
end
TRAIN.y = dataset.y;
TEST.y = dataset_ts.y; 
[u_bar,y_bar] = InputOutputSeq(dataset);
u = dataset.u;
y = dataset.y;
z = [u;y];
%%
[u_q,z_bar,info_q] = q_learn_io(N,u,y,Ry,Ru,df,u_bar,y_bar);
y_q = InferenceConsig(model,sys,N,u_q,x)

%% Data Test 
[u_bar_t,y_bar_t] = InputOutputSeq(dataset_ts);
u_t = dataset_ts.u;
y_t = dataset_ts.y;
[u_q_t,z_bar_t,~] = q_learn_io(N,u_t,y_t,Ry,Ru,df,u_bar_t,y_bar_t);
%% DRQN 
[u_lstm,K_lstm,info_drqn] = DRQN(TRAIN,TEST,N,z);
%%
y_lstm = InferenceConsig(model,sys,N,u_lstm,x)

%% Compute performance index
J_drqn = value_func(y_lstm.y,u_lstm,Ry,Ru,N);
J_q = value_func(y_q.y,u_q,Ry,Ru,N);
