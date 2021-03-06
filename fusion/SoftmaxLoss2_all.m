function [J,g,H] = SoftmaxLoss2_all( wAll, X, y, k, prior, c2c )
%% SoftmaxLoss2_all
%  Desc: get softmax cost, gradient and hessian
%  In: 
%    wAll -- nCluster cell * (nVars{ t } + 1 * nClass{ t } - 1 ), softmax weight
%    X    -- nCluster cell * (nSample * nVars{ t } + 1 ), input feature
%    y    -- nSample * 1, class label for each sample
%    k    -- nCluster cell, each cluster's class number
%    prior-- nSample * nCluster, cluster prior probability for each sample
%    c2c  -- nClass * nCluster, class <--> cluster indicator matrix
%  Out:
%    J    -- cost
%    g    -- gradient
%    H    -- Hessian
%%

nClass   = size( c2c, 1 );
nCluster = size( c2c, 2 );
nSample  = length( y );

wLoc = 1;
wIdx = cell( 1, nCluster );

for t = 1 : nCluster
  [ n{ t }, p{ t } ] = size( X{ t } );
  w{ t } = wAll( wLoc : ( wLoc + p{ t } * ( k{ t } - 1 ) ) - 1 );
  w{ t } = reshape( w{ t }, [ p{ t }, k{ t } - 1 ] );
  w{ t }( :, k{ t } ) = zeros( p{ t }, 1 );    % avoid reduntant weight
  Z{ t } = sum( exp( X{ t } * w{ t } ), 2 );
  wIdx{ t } = wLoc;
  wLoc = wLoc + p{ t } * ( k{ t } - 1 );
end

% class probability for each sample
prob = zeros( nSample, nClass );
for t = 1 : nCluster
  clsIdx = ( c2c( :, t ) == 1 );
  a = bsxfun( @times, exp( X{ t } * w{ t } ), prior( :, t ) );
  prob( :, clsIdx ) = prob( :, clsIdx ) + bsxfun( @rdivide, a, Z{ t } );
end

% cost
ind = sub2ind( size( prob ), ( 1 : nSample )', y );
J = - sum( log( prob( ind ) ) );

% set each sample probability
sampleProb = prob( ind );

% get sample <--> cluster subclass index
s2c = zeros( nSample, nCluster );
for t = 1 : nCluster
  clsIdx = find( c2c( :, t ) == 1 );
  [ ~, s2c( :, t ) ] = ismember( y, clsIdx );
  % set 0 to the last zero weight
  zeroIdx = ( s2c( :, t ) == 0 );
  s2c( zeroIdx, t ) = k{ t };
end


if nargout > 1
  % gradient
  tID = tic;
  g = cell( 1, nCluster );
  for t = 1 : nCluster
    g{ t } = zeros( p{ t }, k{ t } - 1 );
    xt = X{ t };
    wt = w{ t };
    
    frac = bsxfun( @rdivide, prior( :, t ), sampleProb );
    curW =  wt( :, s2c( :, t ) );
    v = sum( exp( xt * wt ), 2 );
    % gradient fraction
    frac = - frac ./ ( v .* v );
    for c = 1 : k{ t } - 1
      u = exp( sum( bsxfun( @times, xt, curW' ), 2 ) );
      isSame = ( s2c( :, t ) == c );
      du = bsxfun( @times, bsxfun( @times, u, isSame ), xt )';
      dv = bsxfun( @times, exp( xt * wt( :, c ) ), xt )';
      a1 = bsxfun( @times, du, v' );
      a2 = bsxfun( @times, u', dv );
      g{ t }( :, c ) = sum( bsxfun( @times, frac', ( a1 - a2 ) ), 2 );
    end % end for each sub class
    % reshape gradient
    g{ t } = reshape( g{ t }, [ p{ t } * ( k{ t } - 1 )  1 ] );
  end % end for each cluster
  g = cat( 1, g{ : } );

  PrintTab(); fprintf( 'compute gradient time -- %.2f (s)\n', toc( tID ) );

end % end if nargout > 1

if nargout > 2
  % Hessian
  tID = tic;
  H = zeros( length( wAll ) );

  for t = 1 : nCluster
    clsIdx = find( c2c( :, t ) == 1 );
    for m = 1 : nSample
      frac = prior( m, t ) ./  sampleProb( m );
      if( s2c( m, t ) == 0 )
        curW = zeros( p{ t }, 1 );
      else
        curW =  w{ t }( :, s2c( m, t ) );
      end
      for c1 = 1 : k{ t } - 1
        % hessian index
        s1 = wIdx{ t } + p{ t } * ( c1 - 1 );
        e1 = wIdx{ t } + p{ t } * c1 - 1;
        [ u, du1 ] = UFunc( X{ t }( m, : ), curW, c1 == s2c( m, t ) );
        [ v, dv1 ] = VFunc( X{ t }( m, : ), w{ t }, c1 );
        v_2 = v .^ 2;
        v_4 = v .^ 4;
        for c2 = 1 : k{ t } - 1
          % hessian index
          s2 = wIdx{ t } + p{ t } * ( c2 - 1 );
          e2 = wIdx{ t } + p{ t } * c2 - 1;
          [ ~, ~, hu ] = UFunc2( X{ t }( m, : ), curW, c1 == s2c( m, t ), c2 == c1 );
          [ ~, dv2 ] = VFunc( X{ t }( m, : ), w{ t }, c2 );
          [ ~, ~, hv ] = VFunc2( X{ t }( m, : ), w{ t }, c1, c2 );
          a1 = v_2 .* ( v .* hu + du1 * dv2' - dv2 * dv1' + v.* hv );
          a2 = 2 .* v .* ( v .* du1 - u .* dv1 ) * dv2';
          H( s1 : e1, s2 : e2  ) =  H( s1 : e1, s2 : e2  ) -  ...
            frac .* ( a1 - a2 ) ./ ( v_4 );
        end % end for c2
      end % end for c1
    end % end for each sample
  end % end for each cluster

  PrintTab(); fprintf( 'compute Hessian time -- %.2f (s)\n', toc( tID ) );

end % end if nargout > 2

end % end function SoftmaxLoss2_all

%-------------------------------
% get u function and derivative
% In:
%   x -- 1 * p vec
%   w -- p * 1 vec
%   isSame -- flag for numerator
% Out:
%   u -- func value
%   du -- p * 1 func derivative
function [ u, du ] = UFunc( x, w, isSame )
  u  = exp( x * w );
  if( isSame )
    du = ( u .* x )';
  else    
    du = ( 0 .* x )';
  end
end % end function UFunc

%-------------------------------
% get v function and derivative
%   x -- 1 * p vec
%   w -- p * k vec
%   c -- subclass index
% Out:
%   u -- func value
%   du -- p * 1 func derivative
function [ v, dv ] = VFunc( x, w, c )
  v  = sum( exp( x * w ), 2 );
  dv = ( exp( x * w( :, c ) ) .* x )';
end % end function VFunc

%-------------------------------
% get u function, first, second derivative
% In:
%   x -- 1 * p vec
%   w -- p * 1 vec
%   isSame -- flag for numerator
% Out:
%   u -- func value
%   du -- p * 1 func derivative
%   hu -- p * p second derivative
function [ u, du, hu ] = UFunc2( x, w, isSame, sameSec )
  p = size( x, 2 );
  u  = exp( x * w );
  if( isSame )
    du = ( u .* x )';
  else    
    du = zeros( p, 1 );
  end
  if( sameSec )
    hu = du * du';
  else
    hu = zeros( p, p );
  end
end % end function UFunc

%-------------------------------
% get v function, first, second derivative
%   x -- 1 * p vec
%   w -- p * k vec
%   c -- subclass index
% Out:
%   v  -- func value
%   dv -- p * 1 func derivative
%   hv -- p * p second derivative
function [ v, dv, hv ] = VFunc2( x, w, c1, c2 )
  p = size( x, 2 );
  v  = sum( exp( x * w ), 2 );
  dv = ( exp( x * w( :, c1 ) ) .* x )';
  if( c2 == c1 ) 
    hv = dv * dv';
  else
    hv = zeros( p, p );
  end
end % end function VFunc
 
