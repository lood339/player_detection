clear
close all

load('annotation_1.mat')
annotation = annot;

data_path = '/Users/jimmy/Desktop/images/bmvc17_soccer/DataSet_001/datasets/';

%% crop player and non-player image patches
N = numel(annotation);
player = [];
non_player = [];
patch_size = [64, 32];
step = 10;

for i = [1:step:N]
    bbox = annotation(i).bbox;
    im_name = [data_path annotation(i).ImgName];
    
    im = imread(im_name);
    [positive, negative] = random_sample_example(im, bbox, patch_size);
    player = cat(4, player, positive);
    non_player = cat(4, non_player, negative);    
end

% balance positive and negative exmples
num = min(size(player, 4), size(non_player, 4));
player = player(:,:,:,[1:num]);
non_player = non_player(:,:,:,[1:num]); 

meta.sets = {'train', 'val'};
meta.org_data = 'player, non-player image patches';
meta.image_mean = [123.8271, 108.3675, 82.0878]; % @todo

images.id = [1:num*2];
images.data = zeros(patch_size(1), patch_size(2), 3, num*2, 'single');  
images.label = zeros(1, num*2, 'double');
images.set = zeros(1, num*2, 'double');

% interleave player and non-player patches
index1 = [1:2:2*num];
index2 = [2:2:2*num];

train_num = int32(0.8*num);
images.data(:,:,:,index1) = player;
images.data(:,:,:,index2) = non_player;
images.label(index1) = 2;
images.label(index2) = 1;
images.set([1:train_num]) = 1; % training
images.set([train_num+1:2*num]) = 2; % validation

% subtract mean value
m = [123.8271  108.3675   82.0878];
m = reshape(m, [1, 1, 3]);
im_mean = repmat(m, patch_size(1), patch_size(2), 1);
for i = [1:2*num]
    images.data(:,:,:,i) = images.data(:,:,:,i) - im_mean;
end

imdb.meta = meta;
imdb.images = images;











