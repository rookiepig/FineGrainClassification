% Generate a grid of 0's to begin with.
m = zeros(400, 400, 'uint8');

% Generate 100 random "trees".
numRandom = 100;
linearIndices = randi(numel(m), 1, numRandom);

% Assign a radius value of 1-12 to each tree
m(linearIndices) = randi(12, [numel(linearIndices) 1]);

buffer = false(size(m));
for radius =1:12 % update to actual range
    im_r  = m==radius;
    se    = strel('disk',radius);
    im_rb = imfilter(im_r, double(se.getnhood()));

    buffer = buffer | im_rb;
end


% The imfilter approach
figure;
h  = fspecial( 'average', 50 );
I1 = imfilter( double( buffer ), h );
imshowpair(buffer,I1, 'montage')

% The nlfilter approach
figure;
f = @(x) numel(x(x==1))/numel(x);
I2 = nlfilter(buffer,[50 50],f);
imshowpair(buffer,I2, 'montage')