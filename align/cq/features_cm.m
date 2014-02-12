function feat = features_cm( im, sbin )
%% features_cm
%  Desc: color moment features from chen qiang
%  In: 
%    im   -- input image
%    sbin -- spatial bins
%  Out:
%    feat -- output features
%%


feat_raw = features_cmmex(double(im), sbin);
cell_size = 4;
feat = zeros([size(feat_raw,1)-cell_size+1, size(feat_raw,2)-cell_size+1, cell_size^2*6]);
for i = 1:cell_size
    for j = 1:cell_size
        idx = (i-1)*cell_size + j;
        feat(:,:,(idx-1)*6+1:idx*6) = feat_raw(i:end-cell_size+i, j:end-cell_size+j, :);
    end
end


% end features_cm


