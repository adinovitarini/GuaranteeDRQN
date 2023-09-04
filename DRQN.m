function [u_lstm,K,info] = DRQN(TRAIN,TEST,N,z)
tic
input_train = TRAIN.input*1e-20;
input_test = TEST.input*1e-20;
target_train = TRAIN.target;
target_test = TEST.target;
input_train = input_train(:,1:N);
% input_train = z_bar(:,1:N);
input_test = input_test(:,1:N);
%%
numFeatures = size(input_train,1);
numResponses = size(target_train,1);
numHiddenUnits = 50;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'StateActivationFunction','tanh')
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.5, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'Plots','none');
%% Train LSTM Net
net = trainNetwork(input_train,target_train,layers,options);

time_elapsed = toc
status = StabilityLSTM(net,z)
%% Predict 
net = predictAndUpdateState(net,input_test);
[~,Qhat] = predictAndUpdateState(net,input_test);
%% Estimate the K via mlp
layers = [ ...
    sequenceInputLayer(size(Qhat,1))
    fullyConnectedLayer(size(TRAIN.y,1))
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',2000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.5, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'Plots','none');
%%
netK = trainNetwork(Qhat,TEST.y*1e-10,layers,options);
%%
[~,K] = predictAndUpdateState(netK,Qhat);
%% Evaluate Performance 
predictionError = target_test-Qhat; 
thr = 0.001;
numCorrect = sum(abs(predictionError) < thr);
numValidationImages = numel(target_test);
accuracy = numCorrect/numValidationImages;
%%
% u_lstm = K.*target_test;
u_lstm = K.*TRAIN.y;
info.time = time_elapsed; 
info.status = status;
info.net = net;
info.netK = netK;
info.numCorrect = numCorrect;
info.Qhat = Qhat;