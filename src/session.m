%% Exercise 2 - Recurrent Neural Networks

%% Initial stuff
clc
clearvars
close all

%% 1 - Hopfield Network

% Note: for this exercise, it will be interesting to check what happens if
% you add more neurons, i.e. modify T such as N is bigger.

T = [1 1; -1 -1; 1 -1]';                                                    % N x Q matrix containing Q vectors with components equal to +- 1.
                                                                            % 2-neuron network with 3 atactors
net = newhop(T);                                                            % Create a recurrent HOpfield network with stable points being the vectors from T

A1 = [0.3 0.6; -0.1 0.8; -1 0.5]';                                          % Example inputs
A2 = [-1 0.5 ; -0.5 0.1 ; -1 -1 ]';
A3 = [1 0.5  ; -0.3 -0.4 ; 0.8 -0.6]';
A4 = [0 -0.1 ; 0.1 0 ; -0.5 0.1]';
A5 = [0 0 ; 0 0.1  ; -0.1 0 ]';
A0 = [1 1; -1 -1; 1 -1]';
% Simulate a Hopfield network
% Y_1 = net([],[],Ai);                                                      % Single step iteration    

num_step = 20;                                                              % Number of steps                
Y_1 = net({num_step},{},A1);                                                % Multiple step iteration
Y_2 = net({num_step},{},A2);  
Y_3 = net({num_step},{},A3);  
Y_4 = net({num_step},{},A4);  
Y_5 = net({num_step},{},A5);

%% Now we try with 4 Neurons

T_ = [1 1 1 1; -1 -1 1 -1; 1 -1 1 1]';                                      % N x Q matrix containing Q vectors with components equal to +- 1.
                                                                            % 4-neuron network with 3 atactors
net_ = newhop(T_);                                                          % Create a recurrent HOpfield network with stable points being the vectors from T

A1_ = [0.3 0.6 0.3 0.6; -0.1 0.8 -0.1 0.8; -1 0.5 -1 0.5]';                 % Example inputs
A2_ = [-1 0.2 -1 0.2 ; -0.5 0.1  -0.5 0.1 ; -1 -1 -1 -1 ]';
A3_ = [1 0.5 1 0.5 ; -0.3 -0.4 -0.3 -0.4 ; 0.8 -0.6 0.8 -0.6]';
A4_ = [-0.5 -0.3 -0.5 -0.3 ; 0.1 0.8 0.1 0.8 ; -0.7 0.6 -0.7 0.6]';
% Simulate a Hopfield network
% Y_1 = net([],[],Ai);                                                      % Single step iteration    

num_step_ = 40;                                                             % Number of steps                
Y_1_ = net_({num_step_},{},A1_);                                            % Multiple step iteration
Y_2_ = net_({num_step_},{},A2_);  
Y_3_ = net_({num_step_},{},A3_);  
Y_4_ = net_({num_step_},{},A4_); 

%% Execute rep2.m || A script which generates n random initial points and visualises results of simulation of a 2d Hopfield network 'net'

% Note: The idea is to understand what is the relation between symmetry and
% attractors. It does not make sense that appears a fourth attractor when
% only 3 are in the Target T
close all 

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n = 15;
d = 0.8;
a1 = [linspace(-d,d,n), linspace(-d,d,n)];
a2 = zeros(length(a1),1);
for i = 1:2*n
    if i <= n
        a2(i) = sqrt((d)^2-a1(i)^2);
    else 
        a2(i) = - sqrt((d)^2-a1(i)^2);  
    end   
    a = {[a1(i); a2(i)]};                  % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);     % simulation of the network for 50 timesteps              
    record = [cell2mat(a) cell2mat(y)];   % formatting results  
    start = cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r', 'linewidth', 2); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO', 'linewidth', 2);  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');

%% Execute rep3.m || A script which generates n random initial points for and visualise results of simulation of a 3d Hopfield network net
% Still needed here to modify the initial points

T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n = 100;
d = 0.8;
a1 = [linspace(-d,d,n), linspace(-d,d,n)];
a2_ = [linspace(-d,d,n),linspace(-d,d,n)];
a2 = zeros(length(a1),1);
a3 = zeros(length(a1),1);

for i=1:2*n
    if i <= n
        a3(i) = sqrt((d)^2-a1(i)^2-a2(i)^2);                         % generate an initial point
    else 
        a3(i) = -sqrt((d)^2-a1(i)^2-a2(i)^2);    
    end
    a2(i) = a2_(i);
    a = {[a1(i); a2(i) ; a3(i)]}; 
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r', 'linewidth', 1);  % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO',  'linewidth', 2);  % plot the final point with a green circle
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');

%% Hopdigit

% Note: Above 5 in the level of noise, even with 50000 iterations not a
% good reconstruction it's achieved.

noise = 6;
numiter = 100;

hopdigit_v2(noise,numiter)

%% Neural Network for Time-Series Prediction
% Alternative solutions: 
% seed = 52, no correction of a, no division of parameters
% seed = 60, correction of a, division of pam

seed = 52;
rng(seed);                                                                  % Seeds the random number generator using the nonnegative integer seed     

% Load the data
load lasertrain.dat 
load laserpred.dat
clear net
close all

p = 18;                                                                     % Lag    
n = 4;                                                                      % Difference between the inputs and the number of neurons    

[TrainData, TrainTarget] = getTimeSeriesTrainData(lasertrain, p);           % This creates TrainData and TrainTarget, such as divided in time  intervales of 5. TrainData(:,1) -> TrainTarget(1). R5 --> R1

% Create the ANN
numN = p - n;                                                               % Number of neurons in the hidden layer
numE = 200;                                                                 % Number of epochs
trainAlg = 'trainlm';                                                       % Training Algorithm 
transFun = 'tansig';                                                        % Transfer Function   

net = feedforwardnet(numN, trainAlg);                                       % Create general feedforward netowrk
net.trainParam.epochs = numE;                                               % Set the number of epochs for the training 
net.layers{1}.transferFcn = transFun;                                       % Set the Transfer Function
% net.divideParam.trainRatio = 0.8;
% net.divideParam.valRatio = 0.2;
% net.divideParam.testRatio = 0;

P = TrainData;       
T = TrainTarget;     
net = train(net,P,T);

TotMat = [lasertrain ; zeros(size(laserpred))];

for j = 1:length(laserpred) 

    r = length(lasertrain) - p + j;
    P = TotMat(r:r+p-1);

    a = sim(net,P);
    TotMat(length(lasertrain) + j) = a;
%     TotMat(length(lasertrain) + j) = (abs(round(a))+round(a))/2;

end
err = immse(TotMat(length(lasertrain)+1:end),laserpred);

rmse = sqrt(mean((TotMat(length(lasertrain)+1:end)' - laserpred').^2)); 
formatSpec = 'The mean squared error is %4.2f \n';
fprintf(formatSpec, err)

x = linspace(1,length(TotMat),length(TotMat));

figure
plot(x(1:length(lasertrain)), TotMat(1:length(lasertrain)),'r', x(length(lasertrain)+1:end), laserpred','b', x(length(lasertrain)+1:end), TotMat(length(lasertrain)+1:end),'g');
xlim([1 1100])
title('Training and test sets with forecasted data');
legend('Training Data','Test Data', 'Predicted data');

figure
subplot(2,1,1)
plot(x(length(lasertrain)+1:end), TotMat(length(lasertrain)+1:end),'g', x(length(lasertrain)+1:end), laserpred','b');
xlim([1000 1100])
title('Predicted data vs Test data');
legend('Predicted Data', 'Test data');

subplot(2,1,2)
stem(TotMat(length(lasertrain)+1:end)' - laserpred')
ylabel("Error")
title("RMSE = " + rmse)

% If you predict a drop is a good set
%% Long short-term memory network

% Initial stuff
close all
clear vars
seed = 0;
rng(seed);                                                                  % Seeds the random number generator using the nonnegative integer seed     

% Load sequence data
load lasertrain.dat 
load laserpred.dat

dataTrain = lasertrain;
numTimeStepsTrain = numel(dataTrain);
dataTest = laserpred;

% Standarize data
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;                             % For a better fit and to prevent the training from diverging 

% Prepare predictors and responses
p = 20;                                                                     % Lag    
[XTrain, YTrain] = getTimeSeriesTrainData(dataTrainStandardized, p);        % This creates TrainData and TrainTarget, such as divided in time  intervales of 5. TrainData(:,1) -> TrainTarget(1). R5 --> R1


% Define LSTM Network Architecture
n = 4;                                                                      % Difference between the inputs and the number of neurons in the hidden layer        
numFeatures = p;
numResponses = 1;
numHiddenUnits = p - n;
epochs = 500;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',epochs, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

% Train LSTM Network
net = trainNetwork(XTrain,YTrain,layers,options);

% Forecast Future Time Steps
dataTestStandardized = (dataTest - mu) / sig;                               % Standardize the test data using the same parameters as the training data
XTest = dataTestStandardized(1:end);

net = predictAndUpdateState(net,XTrain);                                    % Initialize the network state, first predict on the training data XTrain
[net,YPred] = predictAndUpdateState(net,YTrain(end-p+1:end)');              % Make the first prediction using the last time step of the training response YTrain(end)

numTimeStepsTest = numel(XTest);                                            % Loop over the remaining predictions and input the previous prediction   
for i = 2:numTimeStepsTest
    if numel(YPred) < p
        Y_Prev = [ YTrain(end-p+i:end)' ; YPred'];
        [net,YPred(:,i)] = predictAndUpdateState(net, Y_Prev,'ExecutionEnvironment','cpu');
    else  
        [net,YPred(:,i)] = predictAndUpdateState(net,YPred(i-p:end)','ExecutionEnvironment','cpu');
    end
end

YPred = sig*YPred + mu;                                                     % Unstandardize the predictions using the parameters calculated earlier

YTest = dataTest(1:end)';
rmse = sqrt(mean((YPred-YTest).^2));                                        % Calculate the RMSE from the unstandardized predictions.

figure                                                                      % Plot the training time series with the forecasted values
plot(dataTrain(1:end-1))
hold on
plot(linspace(1000,1100,100), YTest) 
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[lasertrain(numTimeStepsTrain) YPred],'.-')
hold off
xlim([1 1100])
legend(["Observed" "Test" "Forecast"])

figure                                                                      % Compare the forecasted values with the test data
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)
