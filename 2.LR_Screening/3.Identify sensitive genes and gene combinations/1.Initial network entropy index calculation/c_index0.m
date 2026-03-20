clear, clc
delete(gcp('nocreate'))
load prewavelet-k1-R100列-A3.mat

adjecent = maprho;
L = 3; % Sliding window length
m = 10; % Training length
adjecent = logical(adjecent);

%% Data processing
n = size(Rec_time, 2); % Total length
delta_record = []; % Error recording

Rec_time = p_data(Rec_time); % Data preprocessing

% Initialize variables
local_H = cell(1, n - m);  

% Parallel processing setup
c = parallel.cluster.Local();
c.NumWorkers = 20;
parpool(c, 20);

% Progress tracking
dq = parallel.pool.DataQueue;
afterEach(dq, @(x) fprintf('Current iteration: %d\n', x));

% Parallel computation
parfor t = m + 1 : n - L + 1
    warning('off', 'MATLAB:rankDeficientMatrix');
    send(dq, t - m);
    
    Rec_time2 = Rec_time(:, t - m : t - 1 + L);
    C = corr(Rec_time2' + 0.001, "type", "Pearson");
    C(isnan(C)) = 0;
    
    predict_local = zeros(size(Rec_time, 1), L);
    delta_local = zeros(size(Rec_time, 1), 1);
    
    for i = 1 : size(Rec_time, 1)
        index = find(abs(C(i, :)) >= 0.6);
        Rec_time2_related = Rec_time2(index, :);
        
        [predict_local(i, :), RMSE] = refine_Main_ARNN(Rec_time2_related', m, L + 1, find(index == i), size(Rec_time2_related, 1));
        delta_local(i) = RMSE;
    end
    
    delta_record(t - m, :) = delta_local;
    predict_local(predict_local < 0) = 0;
    data = [Rec_time2(:, 1:end - L), predict_local];
    
    local_H_t = zeros(size(adjecent, 1), 1);
    
    for k = 1 : size(adjecent, 1)
        p = [];
        coef = [];
        for i = 1 : sum(adjecent(:, k))
            Adj_Rec_time = Rec_time2(adjecent(:, k), :);
            Adj_predict = predict_local(adjecent(:, k), :);
            Adj_data = data(adjecent(:, k), :);
            for j = 1 : L
                p(i, j) = abs(corr(Adj_Rec_time(i, 1:end - L)', Rec_time2(k, 1:end - L)') - corr([Adj_Rec_time(i, 2:end - L), Adj_predict(i, j)]', [Rec_time2(k, 2:end - L), predict_local(k, j)]'));
                if isnan(p(i, j))
                    p(i, j) = 0;
                end
                coef(j) = abs(var(Rec_time2(k, 1:end - L)) - var([Rec_time2(k, 2:end - L), predict_local(k, j)])) / log(sum(adjecent(:, k)) + 0.001);
            end
        end
        p = p ./ (sum(p) + 0.001);
        p_logp = p .* log(p + 0.001);
        edge = -sum(p_logp .* coef, 2);
        local_H_t(k) = sum(edge);
    end
    
    local_H{t - m} = local_H_t;
end

% Combine results
H = zeros(size(adjecent, 1), n - m-2);
for t = m + 1 : n - L + 1
    H(:, t - m) = local_H{t-m};
end

%% Statistical testing
Ht = sum(H) / size(H, 2);
pt = [];
warning_point = [];
for i = 2 : size(Ht, 2)
    [~, pp] = ttest(Ht(1:i-1), Ht(i));
    if 1 / pp > 20
        warning_point = [warning_point, i - 1];
    end
    pt = [pt, 1 / pp];
end

%% Visualization
figureHandel = setHandel(16, 12);
C0 = [1, 0.850980392156863, 0.184313725490196];
C = [1, 0, 0];
C2 = [0.117647058823529, 0.564705882352941, 1];
area(Ht, 'LineWidth', 2, 'FaceColor', C2, 'EdgeColor', C2, 'FaceAlpha', .3, 'EdgeAlpha', 1);
hold on

stem(warning_point(1), Ht(warning_point(1)), 'Marker', 'o', 'MarkerSize', 15, 'Color', C0, 'LineWidth', 1.5)
stem(warning_point(1), Ht(warning_point(1)), 'filled', 'Marker', 'p', 'MarkerSize', 10, 'Color', C)
set(gca, 'LooseInset', get(gca, 'TightInset'))

%%%%%%%%%%%%%%%%%
% Save figures
savefig(figureHandel, 'entropy-K1-R100列.fig');

%% Initial metrics
H0 = Ht;
t0 = warning_point;

%% Error visualization
save error-L100列.mat delta_record
error = mean(delta_record, 'all');

[X, Y] = meshgrid(1:88, 1:length(name));
Stem3D(X, Y, delta_record')

% savefig(gcf, 'error-K1-L100列.fig');
save init_indext_wavelet-R100列-A3.mat H0 t0 H