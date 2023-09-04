function [u_bar,y_bar] = InputOutputSeq(dataset)
%%
y = dataset.y;
u = dataset.u;
%% Iterate input-output data
N = size(u,2);
m = size(u,1);
p = size(y,1);
u_bar = reshape(u,[1 m*N]);
y_bar = reshape(y,[1 p*N]);
