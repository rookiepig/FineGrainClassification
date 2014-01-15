% combine train and test kernel to one kernel

%load( 'imdb.mat' );
% load( 'kernel.mat' );

sampleNum = length( imdb.clsLabel );
kernel = zeros( sampleNum, sampleNum );
kernelTrain = kernelTrain( :, 2 : end );
kernelTest = kernelTest( :, 2 : end );
train = ( imdb.ttSplit == 1 );
test  = ( imdb.ttSplit == 0 );
kernel( train, train ) = kernelTrain;
kernel( test, train  ) = kernelTest;

save( 'kernel_all.mat', 'kernel', '-v7.3' );
