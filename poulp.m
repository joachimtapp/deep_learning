out=load('output.csv');
col=load('color.csv');
clc,close all

scatter3(out(:,1),out(:,2),out(:,3),5*ones(100000,1),col(:),'.'),hold on

[x z] = meshgrid(-5:0.1:5); % Generate x and y data
y = x; % Generate z data
surf(x, y, z)
caxis([-1,1])