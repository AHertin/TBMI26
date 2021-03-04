function [ LPred ] = kNN(X, k, XTrain, LTrain)
    % KNN Your implementation of the kNN algorithm
    %    Inputs:
    %              X      - Samples to be classified (matrix)
    %              k      - Number of neighbors (scalar)
    %              XTrain - Training samples (matrix)
    %              LTrain - Correct labels of each sample (vector)
    %
    %    Output:
    %              LPred  - Predicted labels for each sample (vector)

    % Add your own code here
    [NxData , dims] = size(X);
    [NxtData, ~] = size(XTrain);
    
    % Neigbour N on row 1 and distance to that neighbour on row 2
    tempND = zeros(NxtData, 2);
    
    neighbourV  = zeros(NxtData, NxData);
    distanceV   = zeros(NxtData, NxData);
    
    idx = 1:NxtData;

    for i = 1:NxData
        
        xTemp = X(i, :);
        
        tempND(:, 1) = idx;
        if dims > 2
            for j = 1:dims
                 tempND(:, 2) = tempND(:, 2) + ( XTrain(:,j) - xTemp(j) ).^2;
            end
            tempND(:, 2) = sqrt(tempND(:, 2));
        else
            tempND(:, 2) = sqrt( (XTrain(:,1) - xTemp(1)).^2 + (XTrain(:,2) - xTemp(2)).^2 );
        end
        % Sort tempND by distance to neighbour
        tempND = sortrows(tempND, 2);
        
        neighbourV(:, i)  = tempND(:, 1);
        distanceV(:, i)   = tempND(:, 2);
        
    end
    
    % Set LPred to the k first sorted neigbours
    LPred = LTrain(neighbourV(1:k, :));
    
    if k ~= 1
        LPred = labelChoice(LPred, distanceV(1:k, :));
    end

end

