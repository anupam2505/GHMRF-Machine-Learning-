% Image synthesis using GHMRF
%%
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
fprintf('Performing k-means segmentation\n');
[X, mu, sigma]=img_kmeans(Y,k);
imwrite(uint8(X*120),'initial labels.png');

fprintf('Performing HMRF segmentation\n');
sigmaH = [0.5;0.5;0.5];
[X, mu, sigma]=HMRF_EM(X,Y,Z,mu,sigmaH,k,EM_iter,MAP_iter);
X= mat2gray(X);
%imshow(X)
%imwrite(X,'final labels.png');
figure(2); imagesc(X); colormap(gray); 
drawnow
imwrite(uint8(X*120),'final labels.png');
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