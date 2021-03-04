function [ cM ] = calcConfusionMatrix( LPred, LTrue )
    % CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels
    totLabels = max(LTrue(:,1));
    
    cM = zeros(totLabels, totLabels);
        
    L = length(LTrue(:,1));
    for i = 1:L
        cM(LPred(i,1), LTrue(i,1)) = cM(LPred(i,1), LTrue(i,1)) + 1;
    end
    
%     figure(2001)
%     confusionchart(cM)

end

