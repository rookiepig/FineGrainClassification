%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: step3_fusion.m
% Desc: fusion all group models
% Author: Zhang Kang
% Date: 2014/01/03
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PrintTab();fprintf( 'Run: %s\n', mfilename );

% init configuration
conf = InitConf( );
disp( conf );
load( conf.imdbPath );

if( ~exist( conf.grpInfoPath, 'file' ) || ...
    ~exist( conf.grpModelPath, 'file' ) )
    PrintTab();fprintf( 'Error: grpInfo or grpModel not exist\n' );
else
  % fusion all group models
  load( conf.grpInfoPath );
  load( conf.grpModelPath );
  fusion = GroupFusion( conf, imdb, grpInfo, grpModel );

  % save fusion results and configurations
  save( conf.fusionPath, 'fusion' );
  save( conf.confPath, 'conf' );
  
  %save confusion matrix to PDF file
  meanAccuracy = sprintf('mean accuracy: %.2f %%\n', ...
    100 * mean(diag(fusion.confusion)));
  figure(1) ; clf ;
  imagesc(fusion.confusion) ; axis square ;
  title(meanAccuracy) ;
  vl_printsize(1) ;
  print('-dpdf', fullfile( conf.outDir, [ conf.prefix, '-confusion.pdf' ] ) );
  %print('-djpeg', fullfile(conf.outDir, 'result-confusion.jpg')) ;

end

% end script step3_fusion