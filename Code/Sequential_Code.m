clear;clc;close all;
%Sequential Code

%%
% Image synthesis using FGM model
%
% Random Image generation 
mu = [1;0;0.5];
sigma = [1];
p = [0.33,0.33,0.33];
obj = gmdistribution(mu,sigma,p);
Y = random(obj,512*512);
Y = reshape(Y,512, 512);
figure(1);imshow(Y);
Y=double(Y);


%% Image synthesis using Gibbs sampler and GHMRF
%Realization of Binary Markov Random Field by Gibbs
clear all
close all
%-----------------figure defaults
lw = 2;
set(0, 'DefaultAxesFontSize', 16);
fs = 14;
msize = 5;
randn('state',3) %set the seeds (state) to have
rand ('state',3) %the constancy of results
%----------------
pixelX = 256;
pixelY = 256;
beta = 1;
el = 0;
F = randsrc(256,256, [-1,1]);
for jj=1:50
for k = 1 : 10000
% Select a pixel at random
ix = ceil( pixelX * rand(1) );
iy = ceil( pixelY * rand(1) );
Fc = F( iy, ix );
pos = ( ix - 1 ) * pixelY + iy; % Univariate index of pixel
neighborhood = pos + [-1 1 -pixelY pixelY]; % Find indicies of neighbours
neighborhood( find( [iy == 1 iy == pixelY ix == 1 ix == pixelX] ) ) = [];
% Problematic boundaries...
potential = sum( F(neighborhood) );
i = rand(1);
if i < (exp( - beta * potential )/( exp( - beta * potential )...
+ exp( beta * potential )))
    F( iy, ix ) = -1;
%     if (rand(1) < 10*(exp( - beta * potential )/( exp( - beta * potential )...
% + exp( beta * potential ))))
%     F( iy, ix ) = -1;
%     else
%         F( iy, ix ) = 0;
%     end
% elseif (i > 0.66*(exp( - beta * potential )/( exp( - beta * potential )...
% + exp( beta * potential )))) && (i < 5*(exp( - beta * potential )/( exp( - beta * potential )...
% + exp( beta * potential ))))
% F( iy, ix ) = 0;
else
F( iy, ix ) = 1;
end
el = el + 1;
end
figure(2); imagesc(F); colormap(gray); title(['Iteration for pixels # ' num2str(el)]);
drawnow
end


%% Image sysnthesis for more than 2 class
% Image synthesis using GHMRF
mu = [1;0;0.5];
sigma = [0.02];
p = [0.33,0.33,0.33];
obj = gmdistribution(mu,sigma,p);

Y = random(obj,512*512);
Y = reshape(Y,512, 512);

% Y = rand (512);
% Y = reshape(Y,512, 512);
Z = edge(Y,'canny',0.75);
Y=double(Y);
Y=gaussianBlur(Y,3);

k=3;
EM_iter=2; % max num of iterations
MAP_iter=3; % max num of iterations

tic;
[X, mu, sigma]=img_kmeans(Y,k);

fprintf('Performing Gaussian HMRF segmentation\n');
sigmaH = [0.5;0.5;0.5];
[X, mu, sigma]=HMRF_EM(X,Y,Z,mu,sigmaH,k,EM_iter,MAP_iter);
X= mat2gray(X);
%imshow(X)
%imwrite(X,'final labels.png');
figure(3); imagesc(X); colormap(gray); 
drawnow
toc;


%% NAtural image synthesis using GHMRF


mex BoundMirrorExpand.cpp;
mex BoundMirrorShrink.cpp;

I=imread('/Users/anupampanwar/Google Drive/Studies/FSL/Project/GHMRF-Machine-Learning-/Code/Pascal_Images/boat.jpg');
%I=imread('2SegmentsImage/aeroplane.jpg');
Y=rgb2gray(I);
Z = edge(Y,'canny',0.75);

Y=double(Y);
Y=gaussianBlur(Y,3);

k=3;
EM_iter=10; % max num of iterations
MAP_iter=10; % max num of iterations

tic;
[X, mu, sigma]=img_kmeans(Y,k);

[X, mu, sigma]=HMRF_EM(X,Y,Z,mu,sigma,k,EM_iter,MAP_iter);
figure(2); imagesc(X); colormap(gray); 
drawnow
toc;


% returning the results of kmean along with mean and standard deviation
% used for getting the initial labels of image
function [out m sig]=img_kmeans(in,k)
y=in(:);
x=kmeans(y,k);
out=reshape(x,size(in));
m=zeros(k,1);
sig=zeros(k,1);
for i=1:k
    yy=y(x==i);
    m(i)=mean(yy);
    sig(i)=std(yy);
end
end

% function performing the gaussian blur
function GI=gaussianBlur(I,s)
h=fspecial('gaussian',ceil(s)*3+1,s);
GI=imfilter(I,h,'replicate');
end
