clear all
load('t_wavelet-all-L100列.mat');  % 包含 t_data, point

filename = 'L-averaged_final_adj-k1-100列.xlsx';
if ~exist(filename, 'file')
    error('Excel file not found: %s', filename);
end

[maprho, ~, ~] = xlsread(filename);
Rec_time = t_data;
name = point;

save('prewavelet-L100列.mat', 'Rec_time', 'maprho', 'name');
fprintf('Saved to prewavelet-L100列.mat\n');