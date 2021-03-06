function [X sum_U]=MRF_MAP(X,Y,Z,mu,sigma,k,MAP_iter,show_plot)

[m n]=size(Y);
x=X(:);
y=Y(:);
U=zeros(m*n,k);
sum_U_MAP=zeros(1,MAP_iter);
for it=1:MAP_iter % iterations
    fprintf('  Inner iteration: %d\n',it);
    U1=U;
    U2=U;
    
    for l=1:k % all labels
        yi=y-mu(l);
        temp1=yi.*yi/sigma(l)^2/2;
        temp1=temp1+log(sigma(l));
        U1(:,l)=U1(:,l)+temp1;
        
        
        for ind=1:m*n % all pixels
            [i j]=vec2mat(ind,m);
            u2=0;
            if i-1>=1 && Z(i-1,j)==0
                u2=u2+(l ~= X(i-1,j))/2;
            end
            if i+1<=m && Z(i+1,j)==0
                u2=u2+(l ~= X(i+1,j))/2;
            end
            if j-1>=1 && Z(i,j-1)==0
                u2=u2+(l ~= X(i,j-1))/2;
            end
            if j+1<=n && Z(i,j+1)==0
                u2=u2+(l ~= X(i,j+1))/2;
            end
            U2(ind,l)=u2;
        end
    end
    U=U1+U2;
    [temp x]=min(U,[],2);
    sum_U_MAP(it)=sum(temp(:));
    
    X=reshape(x,[m n]);
    if it>=3 && std(sum_U_MAP(it-2:it))/sum_U_MAP(it)<0.0001
        break;
    end
end

sum_U=0;
for ind=1:m*n % all pixels
    sum_U=sum_U+U(ind,x(ind));
end
if show_plot==1
    figure;
    plot(1:it,sum_U_MAP(1:it),'r');
    title('sum U MAP');
    xlabel('MAP iteration');
    ylabel('sum U MAP');
    drawnow;
end
end


% taking the length of the column (m) and returning the corresponding [i,j] matrix values
% took around 30 mins
% Completed on 20th Nov
function [i j]=vec2mat(val,col)
j=floor((val-1)/col)+1;
i=mod(val-1,col)+1;
end
