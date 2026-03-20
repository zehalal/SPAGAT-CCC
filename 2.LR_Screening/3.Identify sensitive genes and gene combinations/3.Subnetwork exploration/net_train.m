function [Hn0,tn0] = net_train(maprho,Rec_time,m,L)
adjecent = maprho;
adjecent = logical(adjecent);
n = size(Rec_time,2);
predict = [];
delta_record = [];
Rec_time = p_data(Rec_time);

%% 
for t = m+1 : n-L+1
    % t
%ARNN was used to calculate the predicted value of L-day data
    Rec_time2 = Rec_time(:,t-m:t-1+L);
    C = corr(Rec_time2'+0.001,"type","Pearson");
    C(isnan(C))=0;
    for i = 1 : size(Rec_time,1)
        index = find(abs(C(i,:)) >= 0.6);
        Rec_time2_related = Rec_time2(index,:);
        [predict(i,:),RMSE] = refine_Main_ARNN(Rec_time2_related',m,L+1,find(index==i),size(Rec_time2_related,1));
        delta_record(i,t-m) = RMSE;
    end
    predict(predict<0) = 0;
    data = [Rec_time2(:,1:end-L),predict];
% Calculate the entropy of the ARNN network
    for k = 1 : size(adjecent,1)
        p=[];
        coef=[];
        for i = 1 : sum(adjecent(:,k))
                Adj_Rec_time = Rec_time2(adjecent(:,k),:);
                Adj_predict =  predict(adjecent(:,k),:);
                Adj_data = data(adjecent(:,k),:);
            for j = 1 : L
                p(i,j) = abs(corr(Adj_Rec_time(i,1:end-L)',Rec_time2(k,1:end-L)')-corr([Adj_Rec_time(i,2:end-L),Adj_predict(i,j)]',[Rec_time2(k,2:end-L),predict(k,j)]'));
                if isnan(p(i,j))
                    p(i,j)=0;
                end
                coef(j) = abs(var(Rec_time2(k,1:end-L))-var([Rec_time2(k,2:end-L),predict(k,j)]))./log(sum(adjecent(:,k))+0.001);% entropy pre-coefficient
            end
        end 
        p = p./(sum(p)+0.001);
        p_logp = p.*log(p+0.001);
        edge = -sum(p_logp.*coef,2);
        H(k,t-m) = sum(edge);
    end
end
%% Hypothetical sample t-test
Ht = sum(H)/size(H,2);
pt = [];
warning_point=[];
for i = 2 : size(Ht,2)
    [~,pp] = ttest(Ht(1:i-1),Ht(i));
    if 1/pp > 20
        warning_point = [warning_point,i-1];
    end
    pt = [pt,1/pp];
end
%% Calculate the initial metric
Hn0 = Ht;
if isnan(warning_point)
    tn0 = nan;
else
    tn0 = warning_point(1);
end

save results/new_net_init.mat