function metric = apvalumas_roundness(Im)

% converts to greyscale
% greyscale > black n white binary  w threshold 0.95
BW = im2bw(rgb2gray(Im),0.95);

% ~BW: inverts image
% imfill: fills any holes in image
BW = imfill(~BW,'holes');

BW = imopen(BW,strel('disk',12));

BWpr = regionprops(double(BW),{'perimeter','area'});
metric = 4*pi*BWpr(1).Area/BWpr(1).Perimeter^2;