%% EncodeImg: encode imamge using encoder
% -----------------------------------------------------------------
function [ feat ] = EncodeImg( encoder, imgFn, varargin )
% -----------------------------------------------------------------
fprintf( 1, '\n Encode Image ... \n' );
% declare global variables
global conf imdb;

if ( nargin == 3 )
	% 1 paramter --> bounding box list
	curBox = varargin{ 1 };
elseif ( nargin == 4 )
	% 2 paramters --> bounding box + mask list
	curBox = varargin{ 1 };
	maskFn = varargin{ 2 };
end


if( conf.useBoundingBox )
	img = encoder.readImgFunc( imgFn, curBox );
else
	img = encoder.readImgFunc( imgFn ) ;
end

feat = { };
imgSz = size( img );

if( conf.useSegMask )
	mask = encoder.readImgFunc( maskFn, curBox );
  maskType = conf.maskType;
  feat = cell( 1, numel( maskType ) + 1 );
  if( length( find( mask > 0 ) ) > 0.125 * size( mask, 1 ) * size( mask, 2 ) )
        % seg mask occupy > 1/8 area
        for ii = 1 : numel( maskType )
          curMask = ( abs( mask - maskType( ii ) ) <= 1e-4 );
          feat{ ii } = EncodeFeat( encoder, ...
            encoder.getFeatFunc( img, curMask ), imgSz );
        end
      else
        % seg mask too small use whole bounding box 4 times
        fprintf( '\n\t Warning: seg mask too small' );
        for ii = 1 : numel( maskType ) 
            % use the whole bounding box
            feat{ ii } = EncodeFeat( encoder, ...
              encoder.getFeatFunc( img ), imgSz );
          end
        end
    % add bounding box as another feature
    feat{ numel( maskType ) + 1 } = EncodeFeat( encoder, ...
      encoder.getFeatFunc( img ), imgSz );
  else
    % no seg mask just bounding box feature
    feat{ 1 } = EncodeFeat( encoder, encoder.getFeatFunc( img ), imgSz );
  end

  feat = cat( 1, feat{ : } );
  fprintf( '\n\t encoding sparseness: %.2f %%', ...
   100 * length( find( abs( feat ) > 1e-6 ) ) / length( feat )  );

% --------------------------------------------------------------------
function psi = EncodeFeat( encoder, features, imageSize )
% --------------------------------------------------------------------
% encode features from one image

psi = {} ;

for i = 1:size(encoder.subdivisions,2)
  minx = encoder.subdivisions(1,i) * imageSize(2) ;
  miny = encoder.subdivisions(2,i) * imageSize(1) ;
  maxx = encoder.subdivisions(3,i) * imageSize(2) ;
  maxy = encoder.subdivisions(4,i) * imageSize(1) ;

  ok = ...
  minx <= features.frame(1,:) & features.frame(1,:) < maxx  & ...
  miny <= features.frame(2,:) & features.frame(2,:) < maxy ;

  descrs = features.descr(:,ok);
  % each channel has its own PCA
  dimPerCh = conf.featDimPerChannel;
  for pn = 1 : encoder.pcaNum
    desSt = ( pn - 1 ) * dimPerCh + 1;
    desEd = pn * dimPerCh;
    curDescrs = descrs( desSt : desEd, : );
    curDescrs = encoder.projection{ pn } * bsxfun(@minus, ...
      curDescrs, encoder.projectionCenter{ pn } ) ;
    if encoder.renormalize
      curDescrs = bsxfun(@times, curDescrs, 1./max(1e-12, sqrt(sum(curDescrs.^2)))) ;
    end
    tmpDescrs{ pn } = curDescrs;
  end
  descrs = cat( 1, tmpDescrs{ : } );
  
  w = imageSize( 2 );
  h = imageSize( 1 );
  frames = features.frame(1:2,:) ;
  frames = bsxfun(@times, bsxfun(@minus, frames, [w;h]/2), 1./[w;h]) ;

  % currently not support geometry features
  % descrs = extendDescriptorsWithGeometry(encoder.geometricExtension, frames, descrs) ;

  switch encoder.type
  case 'bovw'
  	[words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
  		descrs, ...
  		'MaxComparisons', 100) ;
  	z = vl_binsum(zeros(encoder.numWords,1), 1, double(words)) ;
  	z = sqrt(z) ;

  case 'fv'
  	z = vl_fisher(descrs, ...
  		encoder.means, ...
  		encoder.covariances, ...
  		encoder.priors, ...
  		'Improved') ;
  case 'vlad'
  	[words,distances] = vl_kdtreequery(encoder.kdtree, encoder.words, ...
  		descrs, ...
  		'MaxComparisons', 15) ;
  	assign = zeros(encoder.numWords, numel(words), 'single') ;
  	assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
  	z = vl_vlad(descrs, ...
  		encoder.words, ...
  		assign, ...
  		'SquareRoot', ...
  		'NormalizeComponents') ;
  end
  z = z / max(sqrt(sum(z.^2)), 1e-12) ;
  psi{i} = z(:) ;

end % end for
psi = cat(1, psi{:}) ;

