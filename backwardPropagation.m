function gradients = backwardPropagation(X, Y, cache, n)
    [Z1, A1, w1, b1, Z2, A2, w2, b2] = cache{:,1:8};
    dz2 = A2-Y;
    dw2 = (1/n)*(dz2* A1.');
    db2 = sum(dz2, 2);
    db2 = repelem(db2, 1, n);
    
    dA1 = w2.' * dz2;
    dz1 = dA1.* (A1.*(1-A1));
    dw1 = (1/n) * (dz1* X.');
    db1 = (1/n) * sum(dz1, 2);
    db1 = repelem(db1, 1, n);
    gradients = {dz2, dw2, db2, dz1, dw1, db1};
end