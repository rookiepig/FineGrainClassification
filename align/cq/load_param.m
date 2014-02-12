param.fea_type = 'cm';%'dsift';%'cm';%'dsift','hog','lbp';
param.im_maxsize = 0;%500; % 0 means no rezie for image;
% addpath('classification/tool_features');
% addpath('classification/tool_coding');

switch param.fea_type
    case 'dsift'
        param.multi_scale = [4 6 8 10];
        param.step = 4;
        param.fea_dim = 128;
        param.fea_l2norm = 1;
%         param.multi_scale = [6 10];
%         param.step = 8;
%         param.fea_dim = 128;
%         param.fea_l2norm = 1;
        
        param.isPCA = 1;
        param.PCA_dim = 80;
        
        % add vlfeat path first;
        try
            vl_setup;
        catch
            fprintf('add vlfeat path first;\n');
        end
    case 'hog'
        param.multi_scale = 4:4:30;
        param.fea_dim = 124;
        param.fea_l2norm = 1;
        
        param.isPCA = 1;
        param.PCA_dim = 80;
        
    case 'cm'
        param.multi_scale = 6:2:16;
        param.fea_dim = 96;
        param.fea_l2norm = 0;
        
        param.isPCA = 1;
        param.PCA_dim = 60;
    case 'mnt'
        param.multi_scale = 4:2:10;
        param.fea_dim = 32;
        param.fea_l2norm = 0;
        param.isPCA = 0;
        
    case 'lbp'
        param.multi_scale = 4:4:30;
        param.fea_dim = 59;
        param.fea_l2norm = 1;
        param.isPCA = 0;
end


