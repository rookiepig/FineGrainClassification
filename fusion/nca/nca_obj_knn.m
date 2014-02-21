
%
% wrapper for nca.cc which takes care of multiplying in A and turning
% it into a minimization problem.
%
% idx - K * nSample index matrix
% charless fowlkes
% fowlkes@cs.berkeley.edu
% 2005-02-23
%

function [ F, dF ] = nca_obj_knn( Avec, X, Y, idx )

func = tic;

ID = size(X,2);
A = reshape(Avec,length(Avec)/ID,ID);
[F,dFa] = nca_knn(A,X,Y,A*X',idx);
dF = 2*A*dFa;
dF = -dF(:);
F = -F;

fprintf( 'nca_obj_knn time: %.2f (s)\n', toc( func ) );

% AX = (A*X')';
% perr = class_error((A*X')',Y);
% figure(2); scatter(AX(:,1),AX(:,2),30,perr,'filled'); 
% title('conjgate gradient state');
% drawnow;

