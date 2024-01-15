%% Initialization
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

fprintf('Р вЂ”Р В°Р С–РЎР‚РЎС“Р В¶Р В°Р ВµР С Р С‘ Р Р†Р С‘Р В·РЎС“Р В°Р В»Р С‘Р В·Р С‘РЎР‚РЎС“Р ВµР С Р Т‘Р В°Р Р…Р Р…РЎвЂ№Р Вµ ...\n')

% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

% Plot training data
plotData(X, y);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

%% ==================== Part 2: Training Linear SVM ====================
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.
%

% Load from ex6data1: 
% You will have X, y in your environment
load('ex6data1.mat');

fprintf('\nР С›Р В±РЎС“РЎвЂЎР В°Р ВµР С Р В»Р С‘Р Р…Р ВµР в„–Р Р…РЎвЂ№Р в„– SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 100;
model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
visualizeBoundaryLinear(X, y, model);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

%% =============== Part 3: Implementing Gaussian Kernel ===============
%  You will now implement the Gaussian kernel to use
%  with the SVM. You should complete the code in gaussianKernel.m
%
fprintf('\nР вЂ™РЎвЂ№РЎвЂЎР С‘РЎРѓР В»РЎРЏР ВµР С Р В·Р Р…Р В°РЎвЂЎР ВµР Р…Р С‘РЎРЏ Р С–Р В°РЎС“РЎРѓРЎРѓР С•Р Р†РЎРѓР С”Р С•Р С–Р С• РЎРЏР Т‘РЎР‚Р В° ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Р вЂњР В°РЎС“РЎРѓРЎРѓР С•Р Р†РЎРѓР С”Р С•Р Вµ РЎРЏР Т‘РЎР‚Р С• Р СР ВµР В¶Р Т‘РЎС“ Р Р†Р ВµР С”РЎвЂљР С•РЎР‚Р В°Р СР С‘ x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :' ...
         '\n\t%f\n(Р Т‘Р В»РЎРЏ sigma = 2 РЎРЊРЎвЂљР С• Р В·Р Р…Р В°РЎвЂЎР ВµР Р…Р С‘Р Вµ Р Т‘Р С•Р В»Р Р…Р С• Р В±РЎвЂ№РЎвЂљРЎРЉ Р С—РЎР‚Р С‘Р СР ВµРЎР‚Р Р…Р С• РЎР‚Р В°Р Р†Р Р…Р С• 0.324652)\n'], sigma, sim);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

%% =============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Р вЂ”Р В°Р С–РЎР‚РЎС“Р В¶Р В°Р ВµР С Р С‘ Р Р†Р С‘Р В·РЎС“Р В°Р В»Р С‘Р В·Р С‘РЎР‚РЎС“Р ВµР С Р Т‘Р В°Р Р…Р Р…РЎвЂ№Р Вµ ...\n')

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% Plot training data
plotData(X, y);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nР С›Р В±РЎС“РЎвЂЎР В°Р ВµР С SVM РЎРѓ Р С–Р В°РЎС“РЎРѓРЎРѓР С•Р Р†РЎРѓР С”Р С‘Р С РЎРЏР Т‘РЎР‚Р С•Р С (Р СР С•Р В¶Р ВµРЎвЂљ Р В·Р В°Р Р…РЎРЏРЎвЂљРЎРЉ 1-2 Р СР С‘Р Р…РЎС“РЎвЂљРЎвЂ№) ...\n');

% Load from ex6data2: 
% You will have X, y in your environment
load('ex6data2.mat');

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(X, y, model);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Р вЂ”Р В°Р С–РЎР‚РЎС“Р В¶Р В°Р ВµР С Р С‘ Р Р†Р С‘Р В·РЎС“Р В°Р В»Р С‘Р В·Р С‘РЎР‚РЎС“Р ВµР С Р Т‘Р В°Р Р…Р Р…РЎвЂ№Р Вµ ...\n')

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);

% Train the SVM
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);

fprintf('Р СњР В°Р В¶Р СР С‘РЎвЂљР С‘Р Вµ Enter, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ Р С—РЎР‚Р С•Р Т‘Р С•Р В»Р В¶Р С‘РЎвЂљРЎРЉ.\n');
pause;

