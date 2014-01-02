%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: show_group.m
% Desc: show grouping results by images
% Author: Zhang Kang
% Date: 2013/12/30
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
load( 'imdb.mat' );
load( 'model.mat' );
load( 'grp.mat' );

GRP_SYS_NUM = numel( grp );
for g = 1 : GRP_SYS_NUM
  fprintf( 'Group System: %d\n', g );
  CLUSTER_NUM = numel( grp{ g }.cluster );
  for c = 1 : CLUSTER_NUM
    fprintf( '\t Cluster: %d\n', c );
    P_WID = 5;
    P_HEI = 5;
    plotIdx = 1;
    clsNum = length( grp{ g }.cluster{ c } );
    hFig = figure;
    fprintf( '\t Cur cluster classes: %d\n', clsNum );
    for clsId = 1 : clsNum
      if( clsId > P_HEI )
        break;
      end
      imgIdx = ismember( imdb.clsLabel, grp{ g }.cluster{ c }( clsId ) );
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
