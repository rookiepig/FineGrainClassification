%
% compute the knn stochastic classification error,
% eqn (3) from Goldberger et al NIPS04
%
% for N, D-dimensional data points in K classes
% X should be [N x D], Y should be [N x K]
%
% charless fowlkes
% fowlkes@cs.berkeley.edu
% 2005-02-23
%

function [p_i] = class_error(X,Y)

N = size(X,1);
D = size(X,2);
K = size(Y,2);

r2=sum(X.*X,2);
expD2 = exp(-(repmat(r2,1,N)+repmat(r2',N,1)-2*X*X'));
num = sum(expD2,2) - ones(N,1);
P = expD2./repmat(num,1,N);
P = P - diag(diag(P));

for k = 1:K
  pp(:,k) = sum(P(:,find(Y(:,k))),2);
end;
p_i = sum(pp.*Y,2);

