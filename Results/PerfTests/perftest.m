close all;
clear all;
clc;

mat= [1.4 ,4.959 ,19.8 ,79.9]
totocl = [.820 ,1.304 ,2.824 ,7.684];

x = [128, 256, 512, 1024]
% xtext = [' 128x128 '; ' 256x256 '; ' 512x512 '; '1024x1024'; '2048x2048']
xtext = ['128  '; '  256'; ' 512 '; ' 1024'; ' 2048']

f = figure
plot(x, mat, '-ob','linewidth',2);
hold on
plot(x, totocl, '-og','linewidth',2);

legend('Matlab','OpenCL','FontSize',15,'Location','NorthWest');
title('GPU vs. Matlab runtime','FontSize',20);
xlabel('Image Resolution','FontSize',15);
ylabel('Seconds','FontSize',15);
set(f,'Color','white');
set(gca,'fontsize',15)
set(gca,'XTickLabel',xtext);
grid

set(gca,'XTick',x);
set(gca,'xlim',[50 1050])
%set(gca,'XTickLabel',xtext);

f2 = figure
plot(x, mat./totocl, '-ob','linewidth',2);
hold on;

title('GPU vs. Matlab Speedup','FontSize',20);
xlabel('Image Resolution','FontSize',15);
ylabel('Speedup','FontSize',15);
set(f2,'Color','white');

%plot([0 2048], [1 1])
set(gca,'XTick',x);
set(gca,'xlim',[50 1050])
grid
set(gca,'fontsize',15)
set(gca,'XTickLabel',xtext);