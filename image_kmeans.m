
function [X mu sigma]=image_kmeans(Y,k)
y=Y(:);
x=kmeans(y,k);
X=reshape(x,size(Y));
mu=zeros(k,1);
sigma=zeros(k,1);
for i=1:k
    yy=y(x==i);
    mu(i)=mean(yy);
    sigma(i)=std(yy);
end