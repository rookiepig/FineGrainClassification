%% GetFeat: get dense feature using vl_phow ( can use mask )
% -----------------------------------------------------------------
function [ features ] = GetFeat( img, varargin )
% -----------------------------------------------------------------

% declare global variables
global conf imdb;

[ frame, descr ] = vl_phow( img, conf.featParam{:} );

if( nargin == 2 )
	% mask out unusable descriptors
	mask = varargin{ 1 };
	[ maskY, maskX ] = find( mask > 0 );
	tmpFrame = transpose( round( frame ) );         % round float keypoint
	maskCord = [ maskX maskY ];
	ptIdx = ismember( tmpFrame( :, 1 : 2 ), maskCord, 'rows' );  % get kyepoint index
	descr = descr( :,  ptIdx  );
	frame = frame( :, ptIdx );
end

features.frame = frame;
features.descr = descr;