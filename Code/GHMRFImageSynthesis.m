% Image synthesis using GHMRF
%%

Y = rand (512);
Y = reshape(Y,512, 512);
Z = edge(Y,'canny',0.75);
Y=double(Y);
Y=gaussianBlur(Y,3);

k=3;
EM_iter=10; % max num of iterations
MAP_iter=10; % max num of iterations

tic;
fprintf('Performing k-means segmentation\n');
[X, mu, sigma]=image_kmeans(Y,k);
imwrite(uint8(X*120),'initial labels.png');

fprintf('Performing HMRF segmentation\n');
sigmaH = [0.1;0.1;0.1];
[X, mu, sigma]=HMRF_EM(X,Y,Z,mu,sigmaH,k,EM_iter,MAP_iter);
X= mat2gray(X)
imshow(X)
%imwrite(X,'final labels.png');
imwrite(uint8(X*120),'final labels.png');
toc;