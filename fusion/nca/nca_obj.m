
%
% wrapper for nca.cc which takes care of multiplying in A and turning
% it into a minimization problem.
%
% charless fowlkes
% fowlkes@cs.berkeley.edu
% 2005-02-23
%

function [F,dF] = nca_obj(Avec,X,Y)

ID = size(X,2);
A = reshape(Avec,length(Avec)/ID,ID);
[F,dFa] = nca(A,X,Y,A*X');
dF = 2*A*dFa;
dF = -dF(:);
F = -F;

% AX = (A*X')';
% perr = class_error((A*X')',Y);
% figure(2); scatter(AX(:,1),AX(:,2),30,perr,'filled'); 
% title('conjgate gradient state');
% drawnow;

