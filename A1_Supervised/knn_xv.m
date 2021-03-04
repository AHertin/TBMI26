function [acc,bestK,lPtrain, lPtest] = knn_xv(XTrain, LTrain, XTest, LTest)
    bestK = 0;
    old_acc = 0;
    acc = 0;
    for k = 1:30
        % Classify training data
        LPredTrain = kNN(XTrain, k, XTrain, LTrain);
        % Classify test data
        LPredTest  = kNN(XTest , k, XTrain, LTrain);

        % The confucionMatrix
        cM = calcConfusionMatrix(LPredTest, LTest);

        % The accuracy
        acc = calcAccuracy(cM);

        if acc > old_acc
            bestK = k;
            lPtrain = LPredTrain;
            lPtest  = LPredTest;
        end
        old_acc = acc;

    end
end

