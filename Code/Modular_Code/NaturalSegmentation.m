clear;clc;close all;

mex BoundMirrorExpand.cpp;
mex BoundMirrorShrink.cpp;

I=imread('/Users/anupampanwar/Google Drive/Studies/FSL/Project/GHMRF-Machine-Learning-/Code/Pascal_Images/boat.jpg');
%I=imread('2SegmentsImage/aeroplane.jpg');
Y=rgb2gray(I);
Z = edge(Y,'canny',0.75);

Y=double(Y);
Y=gaussianBlur(Y,3);
%imwrite(uint8(Y),'blurred image.png');

k=3;
EM_iter=10; % max num of iterations
MAP_iter=10; % max num of iterations

tic;
fprintf('Performing k-means segmentation\n');
[X, mu, sigma]=img_kmeans(Y,k);
%imwrite(uint8(X*120),'InitialAeroplaneLabels.png');

[X, mu, sigma]=HMRF_EM(X,Y,Z,mu,sigma,k,EM_iter,MAP_iter);
figure(2); imagesc(X); colormap(gray); 
drawnow
%imwrite(uint8(X*120),'FinalAeroplanelabels.png');
toc;


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

% function performing Gaussian Blur
function GI=gaussianBlur(I,s)
h=fspecial('gaussian',ceil(s)*3+1,s);
GI=imfilter(I,h,'replicate');
end

