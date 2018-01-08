% The MIT License (MIT)
% Copyright (c) 2018 Moritz Nakatenus
% 
% Permission is hereby granted, free of charge, to any person obtaining a 
% copy of this software and associated documentation files (the "Software"), 
% to deal in the Software without restriction, including without limitation 
% the rights to use, copy, modify, merge, publish, distribute, sublicense, 
% and/or sell copies of the Software, and to permit persons to whom the 
% Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
% FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
% DEALINGS IN THE SOFTWARE.

clear;
clc;
close all;

%% SETTINGS
flgDebug = 1;       % print debug msg and plot signals
plotDimId = 1;      % plot this dimension / joint id
numDatasets = 5;       % number of datasets
flgNormalizeTheData = 1;

% plot settings
set(0,'DefaultAxesFontWeight','bold');
set(0,'DefaultAxesFontSize',12);
set(0,'DefaultLineLineWidth', 2);

%% CREATE DATASET
if flgDebug
    NumSamplesUsed = 20; 
end

%% CREATE TRAINING AND TEST SETS
for i=1:numDatasets
    shift = 2 * i;
    datasets{i} = sin(linspace(shift,shift+9,NumSamplesUsed)) + 0 * randn(1,NumSamplesUsed);
end

%% TRAIN THE NETWORK

% initializations
epochs = 50;
trainError = zeros(numDatasets^2-numDatasets, epochs); % test, train, epochs
testError  = zeros(numDatasets, 1);

inputDim  = size(datasets{1},1);
hiddenDim = 10;

% one timestep gets lost because of shifting the data to get input/output relations for the
% network (the output data is the input data shifted by one timestep)
NumSamplesUsed = NumSamplesUsed - 1;

network = lstm_network(NumSamplesUsed, inputDim, hiddenDim, 'Momentum');
        
% cross-validation training process: train on one datasets, test on the
% others
trainCounter = 1;
E = 1:epochs;

%% TEST
for test=1:numDatasets

    trainInd = 1:numDatasets;
    trainInd = trainInd(~ismember(1:numDatasets,test));

    for train=trainInd
    
        %% TRAIN
        X_in = datasets{train}(:,1:end-1);
        %X_in = standardize(X_in); % standardization (zero mean, variance of one)
        Y_out = datasets{train}(:,2:end);
        %Y_out = standardize(Y_out);

        for i=E
            sprintf('Epoch %i',i)

            % train LSTM network with backpropgation through time algorithm
            [Error, Y_pred] = network.BPTT(Y_out, X_in);
            %[Error, Y_pred] = network.TruncatedBPTT(Y_out, X_in,1,10);
            trainError(trainCounter, i) = Error;
            
            % validation-plots
            if flgDebug

                figure(1); clf;
                subplot(1,3,1); plot(Y_pred(plotDimId,:)); xlabel('Timestep'); ylabel('Prediction'); ...
                                    legend(['pred dim ' num2str(plotDimId)]); subplot(1,3,2); ...
                                    plot(Y_out(plotDimId,:)); xlabel('Timestep'); ylabel('Grount Truth');...
                                    legend(['true dim ' num2str(plotDimId)]);
                                    subplot(1,3,3); hold on
                                    plot(Y_out(plotDimId,:),'g'); plot(Y_pred(plotDimId,:),'b'); xlabel('Timestep'); ylabel('Output');...
                                    legend(['true dim ' num2str(plotDimId)],['pred dim ' num2str(plotDimId)]);
            end

        end
        
        trainCounter = trainCounter + 1;

    end

    y = network.forwardPropagation(datasets{test}(:,1:end-1));
    testError(test) = network.mseCost(y, datasets{test}(:,2:end));
end

%% EVALUATE RESULTS
trainMean = mean(trainError,1);
trainStd  = std(trainError,1);

% plot train error with 95% confidence interval
sPos = trainMean + 2 * trainStd;
sNeg = trainMean - 2 * trainStd;

figure(2)
hold all
plot(E, sPos, 'k');
plot(E, sNeg, 'k');
fill([E fliplr(E)], [sNeg fliplr(sPos)], [0.8 0.8 0.8]);
plot(E, trainMean);
xlabel('Epoche')
ylabel('Mean-Squared-Error')
title('MSE Plot with 95% Confidence','fontweight','bold','fontsize',16)

% print test error
sprintf('Test Mean MSE: %d', mean(testError))
sprintf('Test Variance MSE: %d', var(testError))