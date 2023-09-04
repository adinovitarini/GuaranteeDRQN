%%  Check Stability LSTM 
% if norm(Wi) and norm(Wo) < (1-norm(f)_\infty), ...
% norm(Wz)<0.25*1-norm(f)_\infty,...
% norm(f)<(1-norm(f)_\infty)^2,...
% r = O(log(d)) then \phi_lstm is stable
function status = StabilityLSTM(net,z)
%%
T = size(z,2); %number of samples
nh = net.Layers(2).NumHiddenUnits;
Wi = net.Layers(2).InputWeights(1:nh,:);
Wf = net.Layers(2).InputWeights(nh+1:2*nh,:);
Wc = net.Layers(2).InputWeights(2*nh+1:3*nh,:);
Wo = net.Layers(2).InputWeights(3*nh+1:4*nh,:);
% Ri = softmax(net.Layers(2).RecurrentWeights(1:nh,1));
% Rf = softmax(net.Layers(2).RecurrentWeights(nh+1:2*nh,1));
% Rc = softmax(net.Layers(2).RecurrentWeights(2*nh+1:3*nh,1));
% Ro = softmax(net.Layers(2).RecurrentWeights(3*nh+1:4*nh,1));
Ri = (net.Layers(2).RecurrentWeights(1:nh,:));
Rf = (net.Layers(2).RecurrentWeights(nh+1:2*nh,:));
Rc = (net.Layers(2).RecurrentWeights(2*nh+1:3*nh,:));
Ro = (net.Layers(2).RecurrentWeights(3*nh+1:4*nh,:));
h = net.Layers(2).HiddenState;
for i = 1:T  
    inputGate(:,i) = 1./(1+exp((Wi*z(:,i)+Ri*h(:,i))));
    forgetGate(:,i) = 1./(1+exp((Wf*z(:,i)+Rf*h(:,i))));
    cellGate(:,i) = tanh(Wc*z(:,i)+Rc*h(:,i));
    outGate(:,i) = 1./(1+exp((Wo*z(:,i)+Ro*h(:,i))));
    cellGate(:,i) = inputGate(:,i).*forgetGate(:,i).*cellGate(:,i).*outGate(:,i);
    h(:,i+1) = outGate(:,i).*tanh(cellGate(:,i));
end
fk = (supremum(sum(forgetGate,2)));
f_inf = abs(1-fk);
if supremum(sum(Ri,2))<f_inf&&supremum(sum(Ro,2))<=f_inf&&supremum(sum(Rf,2))<(f_inf)^2&&supremum(sum(Rc,2))<f_inf/4
    status = 1;
else
    status = 0;                                                                                                                                                                                                                              
end
end