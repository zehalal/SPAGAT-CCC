function [ax1,ax2,b1,b2]=Butterfly(figureHandle,X1,X2,Label,type)
Y = 1:size(X1,2);
switch type
    case 'stacked'
        ax1 = axes('Parent',figureHandle,'Position',[0.05,0.08,0.5-0.1,0.9]);
        b1 = barh(ax1,Y,X1,0.6,'stacked');
        ax2 = axes('Parent',figureHandle,'Position',[0.55,0.08,0.5-0.1,0.9]);
        b2 = barh(ax2,Y,X2,0.6,'stacked');
    case 'normal'
        ax1 = axes('Parent',figureHandle,'Position',[0.05,0.08,0.5-0.1,0.9]);
        b1 = barh(ax1,Y,X1,0.6);
        ax2 = axes('Parent',figureHandle,'Position',[0.55,0.08,0.5-0.1,0.9]);
        b2 = barh(ax2,Y,X2,0.6);

end
lim = get(ax2,'XLim');
xx = lim(2)/0.4*(-0.05);
for i = 1:length(Y)
    text(xx, i, Label{i}, ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','middle', ...
        'FontSize',9, ...
        'FontName','Arial', ...
        'color','k')
end
end