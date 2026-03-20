%Data preprocessing functions
function Rec_time = p_data(Rec_time)  

for i = 1 : size(Rec_time,1)
    data_r = Rec_time(i,:);
    for j = 3:length(data_r)
        if data_r(j)==0
           data_r(j) = mean(data_r(j-2:j));
        end
        data_r(j) = mean(data_r(j-2:j));
    end
    Rec_time(i,:) = data_r;
end
