function [cost, cache, A2] = forwardPropagation(X, Y, parameters, n)
    [W1, W2, B1, B2] = parameters{:, 1:4};

    Z1 = W1*X + B1;
    A1 = sigmoid(Z1);
    Z2 = W2*A1 + B2;
    A2 = sigmoid(Z2);
    
    cache = {Z1, A1, W1, B1, Z2, A2, W2, B2};
    lo = log(A2).*Y + log(1-A2).*(1-Y);
    cost = -sum(lo) / n;
end