function [X mu sigma]=HMRF_EM(X,Y,Z,mu,sigma,k,EM_iter,MAP_iter)

[m, n]=size(Y);
y=Y(:);
P_lyi=zeros(k,m*n);
sum_U=zeros(1,EM_iter);

for it=1:EM_iter
    fprintf('Iteration: %d\n',it);
    %% update X
    [X, sum_U(it)]=MRF_MAP(X,Y,Z,mu,sigma,k,MAP_iter,0);
    x=X(:);
    %% update mu and sigma
    
    % get P_lyi
    for l=1:k % all labels
        temp1=1/sqrt(2*pi*sigma(l)^2)*exp(-(y-mu(l)).^2/2/sigma(l)^2);
        temp2=temp1*0;
        for ind=1:m*n % all pixels
            [i j]=vec2mat(ind,m);
            u=0;
            if i-1>=1 && Z(i-1,j)==0
                u=u+(l ~= X(i-1,j))/2;
            end
            if i+1<=m && Z(i+1,j)==0
                u=u+(l ~= X(i+1,j))/2;
            end
            if j-1>=1 && Z(i,j-1)==0
                u=u+(l ~= X(i,j-1))/2;
            end
            if j+1<=n && Z(i,j+1)==0
                u=u+(l ~= X(i,j+1))/2;
            end
            temp2(ind)=u;
        end
        P_lyi(l,:)=temp1.*exp(-temp2);
    end
    temp3=sum(P_lyi,1);
    P_lyi=bsxfun(@rdivide,P_lyi,temp3);
    
    % get mu and sigma
    for l=1:k % all labels
        mu(l)=P_lyi(l,:)*y;
        mu(l)=mu(l)/sum(P_lyi(l,:));
        sigma(l)=P_lyi(l,:) * ( (y-mu(l)).^2 );
        sigma(l)=sigma(l)/sum(P_lyi(l,:));
        sigma(l)=sqrt(sigma(l));
    end
    
    if it>=3 && std(sum_U(it-2:it))/sum_U(it)<0.0001
        break;
    end
end

figure;
plot(1:it,sum_U(1:it),'LineWidth',2);
hold on;
plot(1:it,sum_U(1:it),'.','MarkerSize',20);
title('sum of U in each EM iteration');
xlabel('EM iteration');
ylabel('sum of U');
end

% taking the length of the column (m) and returning the corresponding [i,j] matrix values
% took around 30 mins
% Completed on 20th Nov
function [i j]=vec2mat(val,col)
j=floor((val-1)/col)+1;
i=mod(val-1,col)+1;
end