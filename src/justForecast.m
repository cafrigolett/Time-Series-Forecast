
seed = 52
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
%     net.divideParam.trainRatio = 0.8;
%     net.divideParam.valRatio = 0.2;
%     net.divideParam.testRatio = 0;

    P = TrainData;       
    T = TrainTarget;     
    net = train(net,P,T);

    TotMat = [lasertrain ; zeros(size(laserpred))];

    for j = 1:length(laserpred) 

        r = length(lasertrain) - p + j;
        P = TotMat(r:r+p-1);

        a = sim(net,P);
        TotMat(length(lasertrain) + j) = a;
%         TotMat(length(lasertrain) + j) = (abs(round(a))+round(a))/2;

    end
    err = immse(TotMat(length(lasertrain)+1:end),laserpred);
    
    x = linspace(1,length(TotMat),length(TotMat));

figure
subplot(3,1,2)
plot(x(1:length(lasertrain)), TotMat(1:length(lasertrain)),'r', x(length(lasertrain)+1:end), TotMat(length(lasertrain)+1:end),'g', x(length(lasertrain)+1:end), laserpred','b');
xlim([1 1100])
title('Real set and forecast set');
legend('Training Data','Forecasted Data', 'Test data');

subplot(3,1,1)
plot(x(1:length(lasertrain)), TotMat(1:length(lasertrain)),'r', x(length(lasertrain):end), TotMat(length(lasertrain):end),'g');
xlim([1 1100])
title('Forecasted set');
legend('Training Data','Forecasted Data');

subplot(3,1,3)
plot(x(length(lasertrain)+1:end), TotMat(length(lasertrain)+1:end),'g', x(length(lasertrain)+1:end), laserpred','b');
xlim([1000 1100])
title('Detailed forcast data vs Test data');
legend('Forecasted Data', 'Test data');

formatSpec = 'The mean squared error is %4.2f \n';
fprintf(formatSpec, err)
