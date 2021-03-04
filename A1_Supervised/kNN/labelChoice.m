function labelVec = labelChoice(pred, dist)
    % Label choice for when k > 1
    classes = max(unique(pred));
    [k, L] = size(pred);
    
    labelVec = zeros(1, L);
    tempLabel = zeros(1,k);
    tempDist  = zeros(1,k);
    
    for i = 1:L
        tempLabel(:) = pred(:,i);
        tempDist(:)  = dist(:,i);
        minDist = inf;
        [~,~,c] = mode(tempLabel);
        if length(c{1}) == 1
            label = c{1};
        else
            if min(tempDist) == 0
                pos = tempDist == 0;
                label = tempLabel(pos);
            else
                for j = 1:classes
                    labelPos = tempLabel == j;
                    sumDistance = sum(tempDist(labelPos)) / sum(labelPos);
                    if sumDistance < minDist
                        label = j;
                        minDist = sumDistance;
                    end
                end  
            end
        end
        labelVec(i) = label;
    end
    
    labelVec = labelVec';
end

