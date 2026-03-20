function outpng(figureHandle,figureWidth,figureHeight,fileout)
figureUnits = 'centimeters';
figW = figureWidth;
figH = figureHeight;
set(figureHandle,'PaperUnits',figureUnits);
set(figureHandle,'PaperPosition',[0 0 figW figH]);
print(figureHandle,[fileout,'.png'],'-r600','-dpng');