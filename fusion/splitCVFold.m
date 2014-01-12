function [ cvTrain, cvValid ] = SplitCVFold( foldNum, sampleLab, ttSplit )
%% SplitCVFold
%  Desc: split training index to n-fold
%  In: 
%    foldNum -- number of fold
%    sampleLab -- (nSample * 1) sample class label
%    ttSplit -- (nSample * 1) sample training&testing split
%  Out:
%    cvTrain -- (1 * foldNum) train index for each fold
%    cvValid  -- (1 * foldNum) validation index for each fold
%%

PrintTab();fprintf( 'function: %s\n', mfilename );

train  = find( ttSplit == 1 );
nClass = max( sampleLab );
cvTrain = cell( 1, foldNum );
cvValid = cell( 1, foldNum );
for c = 1 : nClass
  trnClsIdx = intersect( find( sampleLab == c  ), train );
  trnClsNum = length( trnClsIdx );
  perm = randperm( trnClsNum );
  patt = ceil( trnClsNum / foldNum );
  for f = 1 : foldNum
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

% end function SplitCVFold
