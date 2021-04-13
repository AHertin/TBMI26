clear variables; close all;
%% Select which data to use:

% 1 = dot cloud 1
% 2 = dot cloud 2
% 3 = dot cloud 3
% 4 = OCR data

dataSetNr = 4; % Change this to load new data 

% X - Data samples
% D - Desired output from classifier for each sample
% L - Labels for each sample
[X, D, L] = loadDataSet( dataSetNr );

%% Select a subset of the training samples

numBins = 4;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

binLine = 1:numBins;

acc = zeros(1,30);

for i = 1:numBins
    XTrain = combineBins(XBins, binLine(1:3));
    LTrain = combineBins(LBins, binLine(1:3));
    XTest  = XBins{binLine(4)};
    LTest  = LBins{binLine(4)};

    acc = acc + knn_xv(XTrain, LTrain, XTest, LTest);
    
    binLine = circshift(binLine,1);
end

acc = acc ./ 4;

[maxAcc, bestK] = max(acc);

%%
LPredT = kNN(XTrain , bestK, XTrain, LTrain);
LPred = kNN(XTest , bestK, XTest, LTest);

%%
    
if dataSetNr < 4
    plotResultDots(XTrain, LTrain, LPredT, XTest, LTest, LPred, 'kNN', [], bestK);
else
    plotResultsOCR(XTest, LTest, LPred)
end
%%
figure(2)
plot(acc)
    
