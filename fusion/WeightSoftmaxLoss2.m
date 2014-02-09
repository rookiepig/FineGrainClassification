function [nll,g,H] = WeightSoftmaxLoss2( w, X, y, k, bias )
% add weight to each sample (bias)

% w(feature*class,1) - weights for last class assumed to be 0
% X(instance,feature)
% y(instance,1)
% bias( instance, 1 ) -- different samples have different weights
% version of SoftmaxLoss where weights for last class are fixed at 0
%   to avoid overparameterization

[n,p] = size(X);
w = reshape(w,[p k-1]);
w(:,k) = zeros(p,1);

Z = sum(exp(X*w),2);

% cost function
nll = -sum( bias .* ( sum(X.*w(:,y).',2) - log(Z) ) );

if nargout > 1
    % gradient
    g = zeros(p,k-1);
    for c = 1:k-1
        g(:,c) = -sum(  repmat( bias, [ 1 p ] ) .* ( X.* repmat( ( y==c ) - exp(X*w(:,c) ) ./ Z, [ 1 p ] )  ) );
    end
    g = reshape(g,[p*(k-1) 1]);
end

if nargout > 2
    % Hessian
    H = zeros(p*(k-1));
    SM = exp(X*w(:,1:k-1))./repmat(Z,[1 k-1]) ;
    for c1 = 1:k-1
        for c2 = 1:k-1
            D = SM(:,c1).*((c1==c2)-SM(:,c2));
            % not sure
            D = D .* bias;
            H((p*(c1-1)+1):p*c1,(p*(c2-1)+1):p*c2) = X'*diag(sparse(D))*X;
        end
    end
end
