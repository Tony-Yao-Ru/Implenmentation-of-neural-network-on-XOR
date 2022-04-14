function parameters = updatePara(parameters, gradients, learningRate)
    [dz2, dw2, db2, dz1, dw1, db1] = gradients{:, 1:6};
    [W1, W2, B1, B2] = parameters{:, 1:4};
    W1 = W1 - learningRate * dw1;
    W2 = W2 - learningRate * dw2;
    B1 = B1 - learningRate * db1;
    B2 = B2 - learningRate * db2;
    parameters = {W1, W2, B1, B2};
end