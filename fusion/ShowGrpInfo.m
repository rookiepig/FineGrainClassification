function ShowGrpInfo( imdb, grpInfo )
%% ShowgrpInfoInfo
%  Desc: show images in each group each cluster (UI needs improve)
%  In: 
%    imdb -- (struct) image database
%    grpInfoInfo -- (struct) group clustering info
%  Out:
%  
%%
fprintf( 'function: %s\n', mfilename );

nGroup = numel( grpInfo );
for g = 1 : nGroup
  fprintf( 'Group System: %d\n', g );
  CLUSTER_NUM = numel( grpInfo{ g }.cluster );
  for c = 1 : CLUSTER_NUM
    fprintf( '\t Cluster: %d\n', c );
    P_WID = 5;
    P_HEI = 5;
    plotIdx = 1;
    clsNum = length( grpInfo{ g }.cluster{ c } );
    hFig = figure;
    fprintf( '\t Cur cluster classes: %d\n', clsNum );
    for clsId = 1 : clsNum
      if( clsId > P_HEI )
        break;
      end
      imgIdx = ismember( imdb.clsLabel, grpInfo{ g }.cluster{ c }( clsId ) );
      clstImgName = imdb.imgName( imgIdx );
      perm = randperm( numel( clstImgName ) );
      selImgName = clstImgName( perm( 1 : P_WID ) );
      for p = 1 : P_WID
        subplot( P_WID, P_HEI, plotIdx );
        imagesc( imread( fullfile( imdb.imgDir, selImgName{ p } ) ) );
        plotIdx = plotIdx + 1;
      end
    end
    set( hFig, 'OuterPosition', [ 200, 200, 800, 800 ] );
    pause;
    fprintf( '\t Press any key to continue..\n' );
    close all;
  end
end
