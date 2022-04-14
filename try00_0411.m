clear
clc
% Define the initial setup including the input and output
Hidenlayer = 2;
epoch = 2000;
learningRate = 0.5;
X = [0, 0, 1, 1; 0, 1, 0, 1];
Y = [0, 1, 1, 0];

% Generate the initial weight and bias
[inputFeature, n] = size(X);
[outputFeature, n] = size(Y);
W1 = randn(Hidenlayer, inputFeature);
W2 = randn(outputFeature, Hidenlayer);
B1 = zeros(Hidenlayer, n);
B2 = zeros(outputFeature, n);
% W1 = [0.5, 0.5; 0.5, 0.5];
% W2 = [0.5, 0.5];
% B1 = [0, 0, 0, 0; 0, 0, 0, 0];
% B2 = [0, 0, 0, 0];

figure
hold on

parameters = {W1, W2, B1, B2};
for i = 1:epoch
    [cost, cache, A2] = forwardPropagation(X, Y, parameters, n);
    plot(i, cost, 'bo')
    drawnow
    gradients = backwardPropagation(X, Y, cache, n);
    parameters = updatePara(parameters, gradients, learningRate);
end

% Test
X_test = [0.0, 1, 1.0, 1.0; 0.0, 1.0, 0.0, 1.0];
[cost, cache, result] = forwardPropagation(X_test, Y, parameters, n);

for i = 1:4
    if result(i) > 0.5
        result(i) = 1;
    else
        result(i) = 0;
    end
end

