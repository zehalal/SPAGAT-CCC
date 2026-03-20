clear, clc

load prewavelet-k1-R100列-A2.mat
load init_indext_wavelet-R100列-A2.mat

delete(gcp('nocreate')); 
% Parallel pool setup
c = parallel.cluster.Local();
c.NumWorkers = 32;
parpool(c, 32);

adjecent = maprho;
L = 3; % sliding window length
m = 10; % training length
adjecent = logical(adjecent);

%% Data processing
Rec_time = p_data(Rec_time);
H_point_value = [];
t_value = [];

% Initialize RMSE storage
RMSE_all = [];

% Perturbation analysis
% for u = [-0.3,-0.25,-0.2,0.2, 0.25, 0.3];
    for u = [-0.3];
    u
    n = size(Rec_time, 2);
    predict = [];
    delta_record = [];
    p_sort_up = ones(size(Rec_time, 1), 1);
    p_sort_down = ones(size(Rec_time, 1), 1);
    t_delta = zeros(size(Rec_time, 1), 1);
    f_delta = zeros(size(Rec_time, 1), 1);
    
    H_local = zeros(size(Rec_time, 1), n - m - L + 1);
    Rec_time_modified = Rec_time;
    delta_record_cell = cell(size(Rec_time, 1), 1);
    
    for nor = 1:size(Rec_time, 1)
        tic;
        if nor>1
          Rec_time_modified(nor-1,:)= temp; 
        end

        temp = Rec_time(nor, :);
        Rec_time_modified(nor, :) = Rec_time_modified(nor, :) + u * Rec_time(nor, :); 
      
        delta_record_all = cell(size(Rec_time, 1), 1);

        parfor t = m + 1 : n - L + 1
            warning('off', 'MATLAB:rankDeficientMatrix');
            Rec_time2 = Rec_time_modified(:, t - m : t - 1 + L);
            C = corr(Rec_time2' + 0.001, "type", "Pearson");
            C(isnan(C)) = 0;
            
            predict_local = zeros(size(Rec_time, 1), L);
            delta_record_local = zeros(size(Rec_time, 1), n - m);
            
            for i = 1:size(Rec_time, 1)
                index = find(abs(C(i, :)) >= 0.6);
                Rec_time2_related = Rec_time2(index, :);
                [predict_local(i, :), RMSE] = refine_Main_ARNN(Rec_time2_related', m, L + 1, find(index == i), size(Rec_time2_related, 1));
                delta_record_local(i, t - m) = RMSE
            end
            
            delta_record_all = [delta_record_all; delta_record_local];
 
            predict_local(predict_local < 0) = 0;
            data = [Rec_time2(:, 1:end - L), predict_local];
            
            H_local_temp = zeros(size(adjecent, 1), 1);
            for k = 1:size(adjecent, 1)
                p = [];
                coef = [];
                
                for i = 1:sum(adjecent(:, k))
                    Adj_Rec_time = Rec_time2(adjecent(:, k), :);
                    Adj_predict = predict_local(adjecent(:, k), :);
                    Adj_data = data(adjecent(:, k), :);
                    
                    for j = 1:L
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
                H_local_temp(k) = sum(edge);
            end
            
            H_local(:, t - m) = H_local_temp;
        end
        
        Ht = sum(H_local) / size(H_local, 2);
        pt = [];
        warning_point = [];
        for i = 2:size(Ht, 2)
            [~, pp] = ttest(Ht(1:i - 1), Ht(i));
            if 1 / pp > 20
                warning_point = [warning_point, i - 1];
            end
            pt = [pt, 1 / pp];
        end
        
        [p_up, p_sort_up(nor)] = ttest(Ht, H0);
        Rec_time(nor, :) = temp;
        
        elapsed_time = toc;
        remaining_time = elapsed_time * (size(Rec_time, 1) - nor) /3600;
 
        fprintf('Gene %d, perturbation %.2f: %.4f sec, remaining: %.2f h\n', nor, u, elapsed_time,remaining_time);
    end

    H_point_value = [H_point_value, p_sort_up];
end

%% Save results
new_name=name;
% StackedButterflyPlot(H_point_value(:, 1:3)', H_point_value(:, 4:6)', new_name);
savefig(gcf, 'final-plot-k1-R100列-A2.fig');
save adduwaveletk1-100-(all)-R-(-0.3)-A2.mat H_point_value;