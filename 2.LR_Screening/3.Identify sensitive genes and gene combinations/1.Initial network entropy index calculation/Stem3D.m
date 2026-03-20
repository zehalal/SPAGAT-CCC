function Stem3D(X,Y,Z)
C1 = [0.741176470588235	0.717647058823529	0.419607843137255];
C2 = [0	1 0];

figureUnits = 'centimeters';
figureWidth = 15;
figureHeight = 12;

figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]);
hold on

%% Three-dimensional stem and leaf drawing
MarkerType = 'o';
MarkerSize = 8;
LineWidth = 1.2;
LineStyle = '-';
st = stem3(X,Y,Z,...
        'MarkerEdgeColor',C1,...      
        'MarkerFaceColor',C2,...      
        'Marker',MarkerType,...      
        'MarkerSize',MarkerSize,...   
        'LineWidth',LineWidth,...     
        'LineStyle',LineStyle,...     
        'Color',C1);                  
hTitle = title('Stem Plot of RMSE');
hXLabel = xlabel('time');
hYLabel = ylabel('gene');
hZLabel = zlabel('RMSE');

%% Detail Optimization
view(-37.5,30)
set(gca, 'Box', 'off', ...                                         
         'XGrid', 'on', 'YGrid', 'on', 'ZGrid', 'on', ...       
         'TickDir', 'out', 'TickLength', [.015 .015], ...        
         'XMinorTick', 'off', 'YMinorTick', 'off', ...           
         'XColor', [.1 .1 .1],  'YColor', [.1 .1 .1])              

set(gca, 'FontName', 'Helvetica')
set([hXLabel, hYLabel, hZLabel], 'FontName', 'AvantGarde')
set(gca, 'FontSize', 10)
set([hXLabel, hYLabel, hZLabel], 'FontSize', 11)
set(hTitle, 'FontSize', 11, 'FontWeight' , 'bold')
set(gcf,'Color',[1 1 1])
set(gca,'LooseInset',get(gca,'TightInset'))
%% Picture output
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
fileout = 'error1';
print(figureHandle,[fileout,'.png'],'-r600','-dpng');