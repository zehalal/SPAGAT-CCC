function [figureHandle]=setHandel(figureWidth,figureHeight)
figureUnits = 'centimeters';
figureHandle = figure;
set(gcf, 'Units', figureUnits, 'Position', [0 0 figureWidth figureHeight]); % define the new figure dimensions
