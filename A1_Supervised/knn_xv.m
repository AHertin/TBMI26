function accV = knn_xv(XTrain, LTrain, XTest, LTest)
    K = 30;
    accV = zeros(1,K);
    for k = 1:K
        % Classify training data
        % LPredTrain = kNN(XTrain, k, XTrain, LTrain);
        % Classify test data
        LPredTest  = kNN(XTest , k, XTrain, LTrain);

        % The confucionMatrix
        cM = calcConfusionMatrix(LPredTest, LTest);

        % The accuracy
        acc = calcAccuracy(cM);

        accV(1, k) = acc;

    end
end

