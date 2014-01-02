%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% File: splitCVFold.m
% Desc: split n-fold cross validataion
% Author: Zhang Kang
% Date: 2014/01/01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf( 'Split %d fold cross validation for fusion\n', conf.foldNum );

% split 10 fold training and validation set
FOLD_NUM = conf.foldNum;
cvTrain = cell( 1, FOLD_NUM );
cvValid = cell( 1, FOLD_NUM );
for c = 1 : numClasses
  trnClsIdx = intersect( find( imdb.clsLabel == c  ), train );
  trnClsNum = length( trnClsIdx );
  perm = randperm( trnClsNum );
  patt = ceil( trnClsNum / FOLD_NUM );
  for f = 1 : FOLD_NUM
    if( c == 1 )
      cvValid{ f } = [];
      cvTrain{ f } = [];
    end
    tmp = ( ( f - 1 ) * patt + 1 ) : min( f * patt, trnClsNum );
    newIdx = trnClsIdx( perm( tmp ) );
    cvValid{ f } = [ cvValid{ f };  newIdx ];
    cvTrain{ f } = [ cvTrain{ f }; setdiff( trnClsIdx, newIdx ) ];
  end
end

fprintf( '\n...Done\n' );
