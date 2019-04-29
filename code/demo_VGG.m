%% This demo shows how to use the REP-YTF protocols and report the results. 
close all; clear; clc;

feaFile = '../data/VGG.mat'; % Mat file storing extracted features for YTF videos. Replace this with your own feature.
imgGalleryFile = '../data/imgGallery_VGG.mat'; % Mat file storing extracted features for gallery images. Replace this with your own feature.
configFile = '../config/ytf/rep_ytf_config.mat'; % configuration file for this evaluation
outDir = '../result/'; % output directory
outLogFile = [outDir, 'result_ytf_VGG.txt']; % output text file

veriFarPoints = [0, kron(10.^(-8:-1), 1:9), 1]; % FAR points for face verification ROC plot
osiFarPoints = [0, kron(10.^(-4:-1), 1:9), 1]; % FAR points for open-set face identification ROC plot
rankPoints = [1:10, 20:10:100]; % rank points for open-set face identification CMC plot
reportVeriFar = 0.001; % the FAR point for verification performance reporting
reportOsiFar = 0.01; % the FAR point for open-set identification performance reporting
reportRank = 1; % the rank point for open-set identification performance reporting
pcaDims = 200; % PCA dimensions
opennessValues = [0.2, 0.5, 0.9]; % openess values for face identification


tic;
fprintf('Load data...\n\n');
load(configFile);

%% Load your own features here. The features should be extracted from each video according
% to the order of the videoList in the configFile. It is 3425xd for the 
% YTF database where d is the feature dimensionality.
Descriptors = importdata(feaFile);

%% Load your own features here. The features should be extracted from each image according
% to the order of the imgGalleryList in the configFile. It is 3425xd for the 
% YTF database where d is the feature dimensionality.
DescriptorsImgGallery = importdata(imgGalleryFile);

numTrials = length(testIndex);
numOpenness = size(galIndex,2);

numVeriFarPoints = length(veriFarPoints);
VR = zeros(numTrials, numVeriFarPoints); % verification rates of the 10 trials
veriFAR = zeros(numTrials, numVeriFarPoints); % verification false accept rates of the 10 trials

numOsiFarPoints = length(osiFarPoints);
numRanks = length(rankPoints);
DIR = zeros(numRanks, numOsiFarPoints, numTrials, numOpenness); % detection and identification rates of the 10 trials for video-to-video scenario
osiFAR = zeros(numTrials, numOpenness, numOsiFarPoints); % open-set identification false accept rates of the 10 trials for video-to-video scenario
RANK = zeros(numRanks, numTrials, numOpenness); % closed-set identification rates for different rank levels of the the 10 trials for video-to-video scenario

DIR_im = zeros(numRanks, numOsiFarPoints, numTrials, numOpenness); % detection and identification rates of the 10 trials for video-to-image scenario
osiFAR_im = zeros(numTrials, numOpenness, numOsiFarPoints); % open-set identification false accept rates of the 10 trials for video-to-image scenario
RANK_im = zeros(numRanks, numTrials, numOpenness); % closed-set identification rates for different rank levels of the the 10 trials for video-to-image scenario

%% Get the FAR or rank index where we report performance.
[~, veriFarIndex] = ismember(reportVeriFar, veriFarPoints);
[~, osiFarIndex] = ismember(reportOsiFar, osiFarPoints);
[~, rankIndex] = ismember(reportRank, rankPoints);

fprintf('Evaluation with 10 trials.\n\n');

%% Evaluate with 10 trials.
for t = 1 : numTrials
    fprintf('Process the %dth trial...\n\n', t);
    
    % Get the training and test data of the t'th trial.
    trainX = Descriptors(trainIndex{t}, :);
    trainLabels = labels(trainIndex{t});
    testX = Descriptors(testIndex{t}, :);
    testLabels = labels(testIndex{t});
    
    % Learn a PCA subspace. Note that if you apply a learning based dimension 
    % reduction or another learning method, it must be performed with the training data of each trial. 
    % It is not allowed to learn and reduce the dimensionality of features
    % with the whole data beforehand and then do the 10-trial evaluation.
    W = PCA(trainX);
    
    % Transform the training and test data into the learned PCA subspace of pcaDims dimensions.
    testX = testX * W(:, 1 : pcaDims);
       
    % Normlize each row to unit length. If you do not have this function,
    % do it manually.
    testX = normr(testX);
    
    % Compute the cosine similarity score between the test samples.
    scores = testX * testX';
         
    % Evaluate the verification performance.
    [VR(t,:), veriFAR(t,:)] = EvalROC(scores, testLabels, [], veriFarPoints);
    
    fprintf('Verification:\n');
    fprintf('\t@ FAR = %g%%: VR = %g%%.\n', reportVeriFar*100, VR(t, veriFarIndex)*100);
    
    for op = 1 : size(galIndex,2) 
    
        % Get the gallery and probe index in the test set
        [~, gIdx] = ismember(galIndex{t,op}, testIndex{t});
        [~, pIdx] = ismember(probIndex{t,op}, testIndex{t});
        
        % Evaluate the open-set and closed-set identification performance for video-to-video scenario.
        [DIR(:,:,t,op), osiFAR(t,op,:), RANK(:,t,op)] = OpenSetROC(scores(gIdx, pIdx), testLabels(gIdx), testLabels(pIdx), osiFarPoints );
        
        fprintf('Open-set Identification: video-video \n');
        fprintf('\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %g%%.\n\n', opennessValues(op), reportRank, reportOsiFar*100, DIR(rankIndex, osiFarIndex, t, op)*100);
        
        % Get the gallery image of the t'th trial for the current op value
        gImg = DescriptorsImgGallery(galIndex{t,op}, :);
        
        % Transform the gallery data into the learned PCA subspace of pcaDims dimensions.
        gImg = gImg * W(:, 1 : pcaDims);
        
        % Normlize each row to unit length. If you do not have this function,
        % do it manually.
        gImg = normr(gImg);
    
        % Compute the cosine similarity score between the gallery and test samples.
        scores_im = gImg * testX(pIdx, :)';
        
        % Evaluate the open-set and closed-set identification performance for video-to-image scenario.
        [DIR_im(:,:,t,op), osiFAR_im(t,op,:), RANK_im(:,t,op)] = OpenSetROC(scores_im, testLabels(gIdx), testLabels(pIdx), osiFarPoints );
               
        fprintf('Open-set Identification: image-video \n');
        fprintf('\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %g%%.\n\n', opennessValues(op), reportRank, reportOsiFar*100, DIR_im(rankIndex, osiFarIndex, t, op)*100);
    end
    clear trainX testX W scores
end

clear Descriptors trainX testX W scores

%% Average over the 10 trials, and compute the standard deviation.
meanVeriFAR = mean(veriFAR);
meanVR = mean(VR);
stdVR = std(VR);
reportMeanVR = meanVR(veriFarIndex)*100;
reportStdVR = stdVR(veriFarIndex)*100;

%% face identification performance 

reportDIR_op = [];
reportRank_op = [];
reportDIR_op_im = [];
reportRank_op_im = [];
reportSTD_op = [];
reportStdRank_op = [];
reportStdDIR_op_im = [];
reportStdRank_op_im = [];

for op = 1 : numOpenness
    
    %%%%%%%%%%%%% video-to-video scenario %%%%%%%%%%%%%%%%%%%%%
    osiFAR_op = osiFAR(:,op,:);
    DIR_op = DIR(:,:,:,op);
    
    meanOsiFAR = mean(osiFAR_op);
    meanDIR = mean(DIR_op, 3);
    stdDIR = std(DIR_op, 0, 3);      
    reportMeanDIR = meanDIR(rankIndex, osiFarIndex)*100;
    reportStdDIR = stdDIR(rankIndex, osiFarIndex)*100;

    % Get the performance measures for open-set
    reportDIR_op = [reportDIR_op, reportMeanDIR];    
    reportSTD_op = [reportSTD_op, reportStdDIR];
    
    RANK_op = RANK(:,:,op)';
    meanRANK = mean(RANK_op);
    stdRANK = std(RANK_op);    
    reportMeanRANK = meanRANK(reportRank)*100;
    reportStdRANK = stdRANK(reportRank)*100;
    
    % Get the performance measures for closed-set
    reportRank_op = [reportRank_op, reportMeanRANK'];
    reportStdRank_op = [reportStdRank_op, reportStdRANK'];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%% video-to-image scenario %%%%%%%%%%%%%%%%%%%%%%%%%%
    osiFAR_op_im = osiFAR_im(:,op,:);
    DIR_op_im = DIR_im(:,:,:,op);
    
    meanOsiFAR_im = mean(osiFAR_op_im);
    meanDIR_im = mean(DIR_op_im, 3);
    stdDIR_im = std(DIR_op_im, 0, 3);      
    reportMeanDIR_im = meanDIR_im(rankIndex, osiFarIndex)*100;
    reportStdDIR_im = stdDIR_im(rankIndex, osiFarIndex)*100;

    % Get the performance measures for open-set
    reportDIR_op_im = [reportDIR_op_im, reportMeanDIR_im];
    reportStdDIR_op_im = [reportStdDIR_op_im, reportStdDIR_im];
    
    RANK_op_im = RANK_im(:,:,op)';
    meanRANK_im = mean(RANK_op_im);
    stdRANK_im = std(RANK_op_im);    
    reportMeanRANK_im = meanRANK_im(reportRank)*100;
    reportStdRANK_im = stdRANK(reportRank)*100;
    
    % Get the performance measures for closed-set
    reportRank_op_im = [reportRank_op_im, reportMeanRANK_im'];
    reportStdRank_op_im = [reportStdRank_op_im, reportStdRANK_im'];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

%% Display the benchmark performance and output to the log file.
str = sprintf('Verification:\n');
str = sprintf('%s\t@ FAR = %g%%: VR = %.2f%% STD = %.2f%%.\n', str, reportVeriFar*100, reportMeanVR, reportStdVR);

str = sprintf('%sOpen-set Identification on video-to-video scenario\n', str);
str = sprintf('%s\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(1), reportRank, reportOsiFar*100, reportDIR_op(1), reportSTD_op(1));
str = sprintf('%s\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(2), reportRank, reportOsiFar*100, reportDIR_op(2), reportSTD_op(1));
str = sprintf('%s\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(3), reportRank, reportOsiFar*100, reportDIR_op(3), reportSTD_op(1));

str = sprintf('%sClosed-set Identification on video-to-video scenario\n', str);
str = sprintf('%s\t@ Op = %g, Rank = %d, IR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(1), reportRank, reportRank_op(1), reportStdRank_op(1));
str = sprintf('%s\t@ Op = %g, Rank = %d, IR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(2), reportRank, reportRank_op(2), reportStdRank_op(1));
str = sprintf('%s\t@ Op = %g, Rank = %d, IR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(3), reportRank, reportRank_op(3), reportStdRank_op(1));

str = sprintf('%sOpen-set Identification on video-to-image scenario\n', str);
str = sprintf('%s\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %.2f%%, STD = %.2f%%\n', str, opennessValues(1), reportRank, reportOsiFar*100, reportDIR_op_im(1), reportStdDIR_op_im(1));
str = sprintf('%s\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %.2f%%, STD = %.2f%%\n', str, opennessValues(2), reportRank, reportOsiFar*100, reportDIR_op_im(2), reportStdDIR_op_im(2));
str = sprintf('%s\t@ Op = %g, Rank = %d, FAR = %g%%: DIR = %.2f%%, STD = %.2f%%\n', str, opennessValues(3), reportRank, reportOsiFar*100, reportDIR_op_im(3), reportStdDIR_op_im(3));

str = sprintf('%sClosed-set Identification on video-to-image scenario\n', str);
str = sprintf('%s\t@ Op = %g, Rank = %d, IR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(1), reportRank, reportRank_op_im(1), reportStdRank_op_im(1));
str = sprintf('%s\t@ Op = %g, Rank = %d, IR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(2), reportRank, reportRank_op_im(2), reportStdRank_op_im(2));
str = sprintf('%s\t@ Op = %g, Rank = %d, IR = %.2f%%, STD = %.2f%%.\n', str, opennessValues(3), reportRank, reportRank_op_im(3), reportStdRank_op_im(3));

fprintf('Overall Performance:\n\n');
fprintf('%s', str);
fout = fopen(outLogFile, 'wt');
fprintf(fout, '%s', str);
fclose(fout);

%% Plot the face verification ROC curve.
figure; semilogx(meanVeriFAR' * 100, meanVR' * 100, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Acceptance Rate (%)');
ylabel('Verification Rate (%)');
title('Face Verification ROC Curve');

%% Plot the open-set face identification ROC curve at the report rank for video-to-video scenario.
figure; semilogx(meanOsiFAR(1,:) * 100, meanDIR(rankIndex,:) * 100, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Acceptance Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set Identification ROC Curve at Rank %d on video-to-video scenario', reportRank));

%% Plot the closet-set face identification CMC curve for video-to-video scenario.
figure; semilogx(rankPoints, meanRANK * 100, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('Rank');
ylabel('Identification Rate (%)');
title('Closed-set Identification CMC Curve on video-to-video scenario');

%% Plot the open-set face identification ROC curve at the report rank for video-to-image scenario.
figure; semilogx(meanOsiFAR(1,:) * 100, meanDIR_im(rankIndex,:) * 100, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('False Acceptance Rate (%)');
ylabel('Detection and Identification Rate (%)');
title(sprintf('Open-set Identification ROC Curve at Rank %d on video-to-image scenario', reportRank));

%% Plot the closet-set face identification CMC curve for video-to-image scenario.
figure; semilogx(rankPoints, meanRANK_im * 100, 'LineWidth', 2);
xlim([0,100]); ylim([0,100]); grid on;
xlabel('Rank');
ylabel('Identification Rate (%)');
title('Closed-set Identification CMC Curve on video-to-image scenario');
