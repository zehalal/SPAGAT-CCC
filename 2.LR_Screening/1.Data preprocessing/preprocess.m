clear all

% Read data table (preserve original column names)
data = readtable('ligand_expr_by_cell.csv', 'Delimiter', ',', 'VariableNamingRule', 'preserve');

% Extract gene names (assuming first column contains gene names)
gene_names = data{:, 1};
point = gene_names;

% Extract numeric data columns
numeric_data = data(:, vartype('numeric'));
data_array = table2array(numeric_data);
result = data_array;

% Data preprocessing
data_r = p_data(result);

% Filter low-expression genes (mean <= 1)
index_0 = mean(data_r, 2) <= 0.2;
data_r(index_0, :) = [];
point(index_0) = [];

% Remove ERCC genes
index_ercc = startsWith(point, 'ERCC');
data_r(index_ercc, :) = [];
point(index_ercc) = [];

% Detect stable intervals
time = detectStableInterval(data_r);
[m, n] = size(data_r);
window_size = 5;
processed_result = data_r;

% Process unstable intervals
if time(1) == 0
    data_temp = p_data(data_r(:, 1:4));
    processed_result(:, 1:4)=data_temp;
end

for k = 2:size(time,2)
    col1 = k * 2 -1;  
    col2 = col1 + 3;
    if time(k) == 0
        data_temp = p_data(data_r(:, col1-2:col2));
        processed_result(:, col1:col2)=data_temp(:,3:end);
    end
end

% Log transform
processed_result = 10 * log(1 + processed_result);

% Interpolate data (5x)
processed_resultnew = zeros(size(processed_result, 1), size(processed_result, 2)*5);
for i = 1:size(processed_result, 1)
    current_row = processed_result(i, :);
    original_x = 1:size(processed_result, 2);
    target_x = linspace(1, size(processed_result, 2), size(processed_result, 2)*5);
    temp = interp1(original_x, current_row, target_x, 'linear');
    processed_resultnew(i, :) = temp;
end

% Select 500 columns via Sobol low-discrepancy sampling for more even coverage
t_data = processed_resultnew;
totalCols = size(processed_result, 2) * 5;
targetCols = 100;
sobol = sobolset(1, 'Skip', 1e3, 'Leap', 1e2);
sobol = scramble(sobol, 'MatousekAffineOwen');
rng(123, 'twister'); % fix seed for reproducibility of scrambling
batchSize = totalCols; % one Sobol point per column for even coverage
idx = unique(floor(net(sobol, batchSize) * totalCols) + 1);

% Fallback: if uniqueness after mapping is insufficient, fill with remaining indices
if numel(idx) < targetCols
    remaining = setdiff(1:totalCols, idx, 'stable');
    idx = [idx; remaining(:)];
end

selectedColumns = sort(idx(1:targetCols));
t_data = t_data(:, selectedColumns);

% Remove all-zero rows and keep point aligned
non_zero_row = any(t_data ~= 0, 2);
t_data = t_data(non_zero_row, :);
point = point(non_zero_row);

% Save results (Malignant ligands dataset)
save t_wavelet-all-ori-R100列2222.mat t_data point

% Prepare Excel output
num_cols = size(t_data, 2);
output_data = cell(size(t_data, 1) + 1, num_cols + 1);
output_data{1, 1} = '';
output_data(1, 2:end) = num2cell(1:num_cols);
output_data(2:end, 1) = point;
output_data(2:end, 2:end) = num2cell(t_data);

