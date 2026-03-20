function rb=calcRb(data,data1)
miu=mean(data(:));
miust=mean(data1(:));
data1afm=abs(data1-miust);
dataafm=abs(data-miu);
rbnumerator=mean(data1afm(:));
rbdenominator=mean(dataafm(:));
rb=rbnumerator/rbdenominator;
end
