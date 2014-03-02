function newFeat = UpdateFeat( oldFeat, preView, curView )
%% UpdateFeat
%  Desc: update allFeat by curView and preView
%  In: 
%    conf - configuration
%    curView - current view
%    preView - previous view
%  Out:
%  
%%
global conf;

nSample      = size( oldFeat, 1 );
VIEW_NUM     = size( conf.viewType, 1 );
TYPE_NUM     = size( conf.viewType, 2 );
ALL_FEAT_LEN = size( oldFeat, 2 );
loc          = zeros( TYPE_NUM, 2 );
part         = ALL_FEAT_LEN / TYPE_NUM;
for v = 1 : TYPE_NUM
  loc( v, 1 ) = ( v - 1 ) * part + 1;
  loc( v, 2 ) = v * part;
end

newFeat = oldFeat;

for n = 1 : nSample
  preFeat = oldFeat( n, : );
  curFeat = preFeat;
  curType = conf.viewType( curView( n ), : );
  preType = conf.viewType( preView( n ), : );
  for t = 1 : TYPE_NUM
    z = find( curType == preType( t ) );
    curFeat( loc( z, 1 ) : loc( z, 2 ) ) = ...
      preFeat( loc( t, 1 ) : loc( t, 2 ) );
  end
  newFeat( n, : ) = curFeat;
end % end for sample


% end UpdateFeat

