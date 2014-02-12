function [fea,pos] = local_fea_extraction(im,param)
% param.fea_type = 'dsift'/'hog'/'cm'/'lbp';
% param.multi_scale = [4 6 8 10]
% param.step = 4;
% pos, dim is the same with matlab version; [width, height]
if ~isfield(param,'fea_type')
    param.fea_type = 'cm';
end

if ~isfield(param,'multi_scale')
    param.multi_scale = [4 6 8 10];
end

if ~isfield(param,'step')
    param.step = 4;
end

switch param.fea_type
    case 'cm'
        if size(im,3)==3
            im=double((im));
        elseif size(im,3)==1
            im = double( cat( 3, im, im, im ) );
        else
            disp('wrong supporting image format!');
            return;
        end
        
        
        fea= {};
        pos= {};
        for i= 1:length(param.multi_scale)
            sbin = param.multi_scale(i);
            temp = features_cm(im,sbin);
            fea{i}= reshape(temp,size(temp,1)*size(temp,2),96)';
            pos{i} = [];
            for m=1:size(temp,1);
                pos{i} = [pos{i},  [ones(1,size(temp,2))*(8+m*sbin);8+(1:size(temp,2))*sbin;]];
            end
        end
        fea = cell2mat(fea);
        pos = cell2mat(pos);
end