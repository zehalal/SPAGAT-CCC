% Clean workspace
clear; clc; close all;
load t_wavelet-all-ori-R100列.mat
m = 126; % Number of genes
n = 100; % Number of samples
exp = t_data;

%% 2. Wavelet transform
wavelet_name = 'db4'; % Daubechies 4 wavelet
level = 5; % Decomposition level

% Initialize trend matrix
trend = zeros(m, n);

% Perform wavelet transform for each gene
for i = 1:m
    [C, L] = wavedec(exp(i, :), level, wavelet_name); % Wavelet decomposition
    trend(i, :) = wrcoef('a', C, L, wavelet_name, level); % Reconstruct approximation coefficients
end

%% 3. Visualization
gene_idx = 1; % Example gene index

figure;
subplot(2, 1, 1);
plot(1:n, exp(gene_idx, :), 'b-', 'LineWidth', 1.5);
xlabel('Sample index');
ylabel('Expression value');
title(['Original data - Gene ', num2str(gene_idx)]);
grid on;

subplot(2, 1, 2);
plot(1:n, trend(gene_idx, :), 'r-', 'LineWidth', 2);
xlabel('Sample index');
ylabel('Trend component');
title(['Trend component - Gene ', num2str(gene_idx)]);
grid on;

% Post-processing
trend(abs(trend) < 0.0001) = 0;
trend = abs(trend);
t_data = trend;

% Drop rows that are all zeros and sync point
non_empty = any(t_data ~= 0, 2);
t_data = t_data(non_empty, :);
point = point(non_empty, :);

% 4. Save results
output_file = 't_wavelet-all-R100列.mat';
save(output_file, 't_data', 'point');
disp(['Trend components saved to: ', output_file]);