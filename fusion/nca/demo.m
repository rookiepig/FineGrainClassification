%
% demonstrate NCA optimization proceedure for finding nice linear projection
%
% charless fowlkes
% fowlkes@cs.berkeley.edu
% 2005-02-23
%

N = 300;
X = [rand(N/2,3); rand(N/2,3)+[0.5*ones(N/2,1) zeros(N/2,1) 1.1*ones(N/2,1)]];
Y = [ones(N/2,1) zeros(N/2,1); zeros(N/2,1) ones(N/2,1)];
[val,class] = max(Y');

sym = 'rgbky';

figure(1); clf; hold on;
for i = 1:size(Y,2)
  ind = find(class == i);
  plot3(X(ind,1),X(ind,2),X(ind,3),[sym(i) '.'],'MarkerSize',12);
end;
hold off;
grid on; axis image;
camorbit(-45,-75);  axis vis3d
title('original 3D data');

figure(2); clf;
A = [1 0 0; 0 1 0];
[Anew,fX,i] = minimize(A(:),'nca_obj',5,X,Y);
Anew = reshape(Anew,2,3);

figure(2); clf;
subplot(2,2,1);
perr = class_error((A*X')',Y);
AX = (A*X')';
scatter(AX(:,1),AX(:,2),30,perr,'filled')
title('initial projection')
subplot(2,2,3); hold on;
for i = 1:size(Y,2)
  ind = find(class == i);
  plot(AX(ind,1),AX(ind,2),[sym(i) '.'],'MarkerSize',12);
end;
hold off

subplot(2,2,2);
perr = class_error((Anew*X')',Y);
AX = (Anew*X')';
scatter(AX(:,1),AX(:,2),30,perr,'filled')
title('final projection')
subplot(2,2,4); hold on;
for i = 1:size(Y,2)
  ind = find(class == i);
  plot(AX(ind,1),AX(ind,2),[sym(i) '.'],'MarkerSize',12);
end;
hold off


