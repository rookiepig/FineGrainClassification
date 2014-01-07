function [ mapFeat ] = NormMapFeat( conf, imdb, mapFeat )
%% NormMapFeat
%  Desc: normalize mapping SVM features
%  In: 
%    conf, imdb -- basic variables
%    mapFeat -- (nSample * nClass) SVM feature with test SVM feat
%  Out:
%    mapFeat -- (nSample * nClass) SVM feature with test SVM feat
%%

fprintf( '\t function: %s\n', mfilename );

% init basic variables
nSample = length( imdb.clsLabel );

% normalize all map features
fprintf( '\t normalize method: %s\n', conf.mapNormType );
for m = 1 : nSample
  z = mapFeat( m, : );
  % find value equal to MAP_INIT_VAl
  initIdx  = find( z == conf.MAP_INIT_VAL );
  otherIdx = find( z ~= conf.MAP_INIT_VAL );
  z( initIdx ) = -1.0;    % set other feature to -1
  switch conf.mapNormType
    case 'l2'
      % l2 normalize 
      z( otherIdx ) = z( otherIdx ) ./ ...
        max( norm( z( otherIdx ), 2 ), 1e-12 );
    case 'l1'
      % l1 normalize
      z( otherIdx ) = z( otherIdx ) ./ ...
        max( norm( z( otherIdx ), 1 ), 1e-12 );
  end

  mapFeat( m, : ) = z;
end

% end function NormMapFeat
