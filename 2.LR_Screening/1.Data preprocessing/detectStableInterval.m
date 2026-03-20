function time = detectStableInterval(data)
% Detects stable time intervals in gene expression data
[r,c] = size(data);
rbmatrix = [];

% Calculate stability ratios for sliding windows
for i = 1:2:(c-2)
    data1 = data(:,i:i+2);
    rbmatrixte = [];
    
    for j = 1:r
        data2 = data1(j,:);
        rb = calcRb(data1,data2);
        rbmatrixte(j,1) = rb;       
    end
    rbmatrix = [rbmatrix,rbmatrixte];
end

% Determine mergeable time intervals
[r,c] = size(rbmatrix);
time = [];

for i = 1:c-1
    x = rbmatrix(:,i);
    y = rbmatrix(:,i+1);
    h = judgeStability(x,y);
    time(1,i) = ~h; % 1 if mergeable, 0 otherwise   
end
end