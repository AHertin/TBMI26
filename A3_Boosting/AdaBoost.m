clear variables; close all

%% Hyper-parameters

rng(666)

% Number of randomized Haar-features
nbrHaarFeatures = 400;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 36;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError

d = ones(1,nbrTrainImages)./nbrTrainImages;

nClass = nbrWeakClassifiers;

train_data = zeros(nClass, 4);

c_out = zeros(nClass, nbrTrainImages);

for i = 1:nClass
    eps_min   = inf;
    cut_min   = 0;
    pol_out   = 0;
    alpha_min = 0;
    feat_min  = 0;
    
    
    for f = 1:nbrHaarFeatures
        
        thresholds = xTrain(f,:);
        
        for cut = thresholds
           
            pol   = 1;
            c     = WeakClassifier(cut, pol, xTrain(f,:));
            eps_t = WeakClassifierError(c, d, yTrain);
            
            if eps_t > 0.5
               
                pol   = -pol;
                eps_t = 1 - eps_t;
                c     = -c;
            end
            
            if eps_t < eps_min
                
                eps_min   = eps_t;
                alpha     = log((1 - eps_min) / eps_min) / 2;
                cut_min   = cut;
                pol_out   = pol;
                feat_min  = f;
                alpha_min = alpha;
                c_min     = c;
                
            end
            
        end
        
    end
    
    if eps_min == 0.5
        break;
    end
    
    d = d.*exp(-alpha_min * yTrain .* c_min);
    d = d./sum(d);
    
    train_data(i,1) = cut_min;
    train_data(i,2) = pol_out;
    train_data(i,3) = feat_min;
    train_data(i,4) = alpha_min;
    c_out(i,:)      = c_min * alpha_min;
    
    
end


%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.

test_eval  = zeros(1, nbrTestImages);
test_acc   = zeros(nClass, 1);
train_eval = zeros(1, nbrTrainImages);
train_acc  = zeros(nClass, 1);

for j = 1:nClass
    test_eval    = test_eval + train_data(j,4) * WeakClassifier(train_data(j,1), train_data(j,2), xTest(train_data(j,3),:));
    test_acc(j)  = sum(sign(test_eval) == yTest)/nbrTestImages;
    train_eval   = train_eval + train_data(j,4) * WeakClassifier(train_data(j,1), train_data(j,2), xTrain(train_data(j,3),:));
    train_acc(j) = sum(sign(train_eval) == yTrain)/nbrTrainImages;
end

strong_class_test = sign(test_eval);
strong_class_train = sign(train_eval);

test_accuracy  = sum(strong_class_test == yTest)/nbrTestImages;
train_accuracy = sum(strong_class_train == yTrain)/nbrTrainImages;

%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.

figure(4)
hold on
plot(test_acc, '-b')
plot(train_acc, '-r')
title('Accuracy')
xlabel('nbr of weak classifiers')
legend('test acc', 'training accuracy')

%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.

wrong_classification = strong_class_test ~= yTest;
wrong_images         = testImages(:,:,wrong_classification);
wrong_class          = strong_class_test(wrong_classification);
wrong_faces          = wrong_images(:,:,wrong_class == -1);
wrong_nonfaces       = wrong_images(:,:,wrong_class == 1);

figure(5)
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(wrong_faces(:,:,10*k));
    axis image;
    axis off;
end


figure(6);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(wrong_nonfaces(:,:,10*k));
    axis image;
    axis off;
end


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.

figure(7);
colormap gray;
sub_size = ceil(sqrt(nClass));
for k=1:nClass
    subplot(sub_size,sub_size,k), imagesc(haarFeatureMasks(:,:,train_data(k,3)));
    axis image;
    axis off;
end

%% Clear temp workspace
clear eps_min cut_min feat_min alpha_min eps_t c_min c pol i j k f pol_out cut alpha
