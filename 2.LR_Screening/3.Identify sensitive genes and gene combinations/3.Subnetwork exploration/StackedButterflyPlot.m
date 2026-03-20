function StackedButterflyPlot(X1,X2,Label)
C = [0.647058823529412	0	0.149019607843137
    0.843137254901961	0.200000000000000	0.152941176470588
    0.960784313725490	0.458823529411765	0.278431372549020
    0.992156862745098	0.721568627450980	0.419607843137255
    0.996078431372549	0.909803921568627	0.615686274509804
    0.952941176470588	0.980392156862745	0.827450980392157
    0.788235294117647	0.905882352941177	0.945098039215686
    0.556862745098039	0.756862745098039	0.866666666666667
    0.345098039215686	0.549019607843137	0.749019607843137
    0.219607843137255	0.298039215686275	0.623529411764706];
C1 = C(1,1:3);
C2 = C(2,1:3);
C3 = C(3,1:3);
C5 = C(10,1:3);
C6 = C(9,1:3);
C7 = C(8,1:3);

figureUnits = 'centimeters';
figureWidth = 16;
figureHeight = 12;

figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);

[ax1,ax2,b1,b2] = Butterfly(figureHandle,X1,X2,Label,'stacked');

b1(1).FaceColor = C1;
b1(2).FaceColor = C2;
b1(3).FaceColor = C3;
set(ax1, 'Box','off',...
         'LineWidth',1,...
         'TickLength',[0 0],...
         'XGrid','on','YGrid','off',...
         'XDir','reverse',...
         'YDir','reverse',...
         'YAxisLocation','right',...
         'YTick',[])
%ax1.XRuler.Axle.LineStyle = 'none'; 
hLegend1 = legend(ax1, ...
                 '-30%','-25%','-20%', ...
                 'Location', 'northoutside',...
                 'Orientation','horizontal');
hLegend1.ItemTokenSize = [10 10];
hLegend1.Box = 'off';

set([ax1,hLegend1], 'FontName', 'Arial', 'FontSize', 9)

b2(1).FaceColor = C5;
b2(2).FaceColor = C6;
b2(3).FaceColor = C7;
set(ax2, 'Box','off',...
         'LineWidth',1,...
         'TickLength',[0 0],...
         'XGrid','on','YGrid','off',...
         'XDir','normal',...
         'YDir','reverse',...
         'YAxisLocation','left',...
         'YTick',[])
ax2.XRuler.Axle.LineStyle = 'none';  
hLegend2 = legend(ax2, ...
                 '30%','25%','20%', ...
                 'Location', 'northoutside',...
                 'Orientation','horizontal');
hLegend2.ItemTokenSize = [10 10];
hLegend2.Box = 'off';
set([ax2,hLegend2], 'FontName', 'Arial', 'FontSize', 9)

set(gcf,'Color',[1 1 1])
set(gca,'LooseInset',get(gca,'TightInset'))
%% Picture output
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'comb_p_1';
print(figureHandle,[fileout,'.png'],'-r600','-dpng');