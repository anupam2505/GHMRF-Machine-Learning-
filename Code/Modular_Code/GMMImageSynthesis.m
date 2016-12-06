% FSL code
% Image segmentation using HMRF model

%%
% Random Image generation using GMM 
mu = [1;0;0.5];
sigma = [1];
p = [0.33,0.33,0.33];
obj = gmdistribution(mu,sigma,p);

Y = random(obj,512*512);
Y = reshape(Y,512, 512);

imshow(Y);

Y=double(Y);
