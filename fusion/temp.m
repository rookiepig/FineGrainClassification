%%
% set non-cluster scores to negative infinite
%
%   testNum = length( test );
%   nClass = max( imdb.clsLabel );
%  
%   for g = 1 : conf.nGroup
%     tryScores =  g1;
%     cTc = grpInfo{ g }.clsToCluster( test );
%     % set other cluster score to minimum
%     for t = 1 : testNum
%       clsIdx = grpInfo{ g }.cluster{ cTc( t ) };
%       clsIdx = setdiff( ( 1 : nClass )', clsIdx );
%       tryScores( t, clsIdx ) = -1e10;
%     end
%     [ tryConf, tryMeanAcc ] = ScoreToConf( tryScores, testLab );
%     fprintf( 'Use Group %d cluster prior -- mA: %f %%\n', g, tryMeanAcc );
%     pause;
%   end
% 
% 
% for ii = 1 : 32
%   if( any( ismember( 128, curGrp.cluster{ ii } ) ) )
%     disp( ii );
%   end
% end
% 
% 
% 
% meanAcc = 0;
% for c = 1 : curGrp.nCluster
%   fprintf( '\t Cluster: %d (%.2f %%)\n', c, 100 * c / curGrp.nCluster );   
%   % use clustering results
%   testIdx = intersect( find( curGrp.clsToCluster == c ), ...
%     test );   
%   grpCls = curGrp.cluster{ c };
%   tmpScore = zeros( length( testIdx ), length( grpCls ) );
%   tmpLabel = imdb.clsLabel( testIdx );
%   tmpGtLabel = floor( rand( size( tmpLabel ) ) * length( grpCls ) + 1 );
%   for gC = 1 : length( grpCls )
%     cmpCls = grpCls( gC );
%     tmpIdx = find( tmpLabel == cmpCls );
%     tmpGtLabel( tmpIdx ) = gC;
%     fprintf( '\t\t train test class: %d\n', cmpCls );
%     tmpScore( :, gC ) = curModel.mapFeat( testIdx, cmpCls );
%   end % end for grpCls
%   [ tmpConf, tmpAcc ] = ScoreToConf( tmpScore, tmpGtLabel );
%   fprintf( '\t cluster mA: %.2f %%\n', tmpAcc );
%   meanAcc = meanAcc + tmpAcc;
%   fprintf( 'Press any key to continue..\n' );
%   pause;
% end % end for cluster
% meanAcc = meanAcc / curGrp.nCluster;
% fprintf( 'All cluster mean mA: %.2f %%\n', meanAcc );

%%
% combine cluster score with SVM score
%
% tmpG = 6;
% nSample = length( imdb.clsLabel );
% newScore = zeros( nSample, 200 );
% for t = 1 : grpInfo{ tmpG }.nCluster
%   grpCls = grpInfo{ tmpG }.cluster{ t };
%   newScore( :, grpCls ) = repmat( grpInfo{ tmpG }.clusterScore( :, t ), ...
%     [ 1, length( grpCls ) ] );
% end
% newFeat = bsxfun( @times, newScore, grpModel{ tmpG }.mapFeat );
% [ ~, newAcc ] = ScoreToConf(  newFeat( test ), testLab );
% newAcc

%% 
% cluster prior

%   testNum = length( test );
%   nClass = max( imdb.clsLabel );
%   for g = 1 : 6
%     cTc = grpInfo{ g }.clusterGtLab( test );
%     newScores = grpModel{ g }.scores;
%     % set other cluster score to minimum
%     for t = 1 : testNum
%       clsIdx = grpInfo{ g }.cluster{ cTc( t ) };
%       clsIdx = setdiff( ( 1 : nClass )', clsIdx );
%       newScores( t, clsIdx ) = -1e10;
%     end
%     [ ~, newAcc ] = ScoreToConf( newScores, testLab );
%     fprintf( 'group %d -- acc %.2f %%\n', g, newAcc );
%   end

%%
%% Multinomial logistic regression with L2-regularization
nInstances = size( trainScore, 1 );
nClasses = 200;
nVars = 200;
options.Display = 1;

X = trainScore;
y = trainLab;
% Add bias
X = [ones(nInstances,1) X];
XAll = [ ones( size( allScore, 1 ), 1 ) allScore ];

funObj = @(W)SoftmaxLoss2(W,X,y,nClasses);
lambda = 1e-4*ones(nVars+1,nClasses-1);
lambda(1,:) = 0; % Don't penalize biases
fprintf('Training multinomial logistic regression model...\n');
wSoftmax = minFunc(@penalizedL2,zeros((nVars+1)*(nClasses-1),1),options,funObj,lambda(:));
wSoftmax = reshape(wSoftmax,[nVars+1 nClasses-1]);
wSoftmax = [wSoftmax zeros(nVars+1,1)];

[junk yhat] = max(X*wSoftmax,[],2);
trainErr = sum(yhat~=y)/length(y)

% Plot the result
% figure;
% plotClassifier(X,y,wSoftmax,'Multinomial Logistic Regression');
% pause;




