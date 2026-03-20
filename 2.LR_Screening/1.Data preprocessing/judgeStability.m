function h=judgeStability(x,y)
% Determines if two samples come from same distribution
[H1,P,LSTAT,CV] = lillietest(x,0.01);
[H2,P,LSTAT,CV] = lillietest(y,0.01);
if H1==0 && H2==0
    [h,p1,ci1]=ttest2(x,y,0.025);
else
    [p,h,stats] = ranksum(x, y,0.01);
end
if h==0
   disp("Normal distribution, merging data")
end
end