clear,clc;
% load results/adduwaveletk1-500-(all)-hESC.mat
% load data/prewavelet-k1-hESC.mat
% load H_point_value_concat-R.mat
% load prewavelet-k1-R.mat

load adduwaveletk1-100-(all)-R-merged.mat
load prewavelet-k1-R100列.mat

% Filter genes below threshold
new_point = find(mean(H_point_value,2)<0.254);
new_name = name(new_point);

% Filter non-zero connections
new_maprho = maprho(new_point,new_point);
nonzero_rows = any(new_maprho, 2);
nonzero_cols = any(new_maprho, 1);
new_maprho = new_maprho(nonzero_rows,nonzero_cols);
new_Rectime= Rec_time(new_point,:);
new_Rectime=new_Rectime(nonzero_rows,:);
new_point=new_point(nonzero_rows);
new_name=new_name( nonzero_rows);

% Train initial network
[Hn0,tn0] = net_train(new_maprho,new_Rectime,10,3);
sample = 1:size(new_Rectime,1);

% Calculate total combinations
n=0;
for k = 1:size(new_Rectime,1)
    n = n + nchoosek(size(new_Rectime,1),k);
end

% Initialize records
u = [-0.3,-0.25,-0.2,0.2,0.25,0.3]; 
C_record = zeros(n,length(u)+size(new_Rectime,1));

% Generate valid combinations
index0 = 1;
for k = 1:size(new_Rectime,1)
    C=nchoosek(sample,k);
    
    % Check connectivity
    for i = 1:size(C,1)
        C_i = C(i,:);
        Ci_map = maprho(C_i,C_i);
        conj = conncomp(graph(Ci_map));
        if ~all(conj==1)
            C(i,:) = zeros(1,size(C,2));
        end
    end
        
    C(sum(C,2)==0,:)=[];
    tempC = [C,zeros(size(C,1),size(new_Rectime,1)-k)];
    C_record(index0:index0+size(C,1)-1,1:size(new_Rectime,1)) = tempC;
    index0 = index0 + size(C,1);
    clear C
end

% Apply perturbations
Hrecord = zeros(size(C_record,1),size(C_record,2)-size(new_Rectime,1),size(Hn0,2));
trecord = C_record;
index_i = 0;

for i = u
    index_i = index_i + 1;
    tic;
    for j = 1:size(C_record, 1)
        fprintf('u = %.2f, j = %d\n', i, j); 
        tempC2 = C_record(j, 1:size(new_Rectime, 1));
        indexC = tempC2(tempC2 ~= 0);
        temp = new_Rectime(indexC', :);
        new_Rectime(indexC', :) = temp + i * temp;       
        [Hrecord(j, index_i, :), trecord(j, size(new_Rectime, 1) + index_i)] = net_train(new_maprho, new_Rectime, 10, 3);
        [~, C_record(j, size(new_Rectime, 1) + index_i)] = ttest(Hn0, reshape(Hrecord(j, index_i, :), 1, []));
        new_Rectime(indexC', :) = temp;
        clear temp tempC2 indexC
    end
    
    elapsedTime = toc/3600;
    fprintf('Iteration u = %.2f finished in %.2f h.\n', i, elapsedTime);
end

% Analyze results
[p_sort,p_index] = sort(mean(C_record(:,size(new_Rectime,1)+1:end),2),'ascend');
best_group = C_record(mean(C_record(:,size(new_Rectime,1)+1:end),2)<0.01,1:size(new_Rectime,1));
best_group_name = num2cell(best_group);
for i = 1:size(best_group,2)
    best_group_name(best_group==i) = new_name(i);
end

delta_trecord = trecord(:,size(new_Rectime,1)+1:end) - tn0;
select_H = Hrecord(p_index(1:3),:,:);

% Visualization
label = C_record(:,1:size(new_Rectime,1));
temp_index = sum(label,2)~=0;
C_plot_record = C_record(:,size(new_Rectime,1)+1:end);
C_plot_record(~temp_index,:)=[];
label(~temp_index,:)=[];
delta_trecord(~temp_index,:)=[];

strlabel =[];
for i = 1:size(label,1)
    temp = num2str(label(i,:));
    strlabel = [strlabel;temp];
end

StackedButterflyPlot(C_plot_record(:,1:3)',C_plot_record(:,4:6)',cellstr(strlabel)')
savefig(gcf, 'results/subnetwork-butterfly-k1-hESC.fig');

% Line plots
target_geneconbination = delta_trecord(mean(C_plot_record,2)<0.01,:);
target_label = strlabel(mean(C_plot_record,2)<0.01,:);
zexianHandel = setHandel(16,12);
C = [0.8941 0.1020 0.1098];

for i = 1:size(target_geneconbination,1)
    subplot(2,3,i)
    plot([-0.3,-0.25,-0.2,0.2,0.25,0.3],target_geneconbination(i,:),'Color',C)
    title(target_label(i,:))
    set(gca,'XGrid','on','XColor',[.1 .1 .1],'YColor',[.1 .1 .1],...
        'XTick',[-0.3:0.1:-0.1,0.1:0.1:0.3],'XTicklabel',[-0.3:0.1:-0.1,0.1:0.1:0.3],...
        'XMinorTick','on','YMinorTick','on')
end

set(gcf,'Color','w');
set(gca,'LooseInset',get(gca,'TightInset'))
savefig(zexianHandel, 'results/final_result-plot-k1-hESC.fig');
outpng(zexianHandel,16,12,'function1')
save results/final_result1_wavelet-k1((1.5e-6)-hESC.mat C_record Hrecord trecord delta_trecord best_group_name Hn0 tn0 new_Rectime new_name C_plot_record