clear
close all

addpath('./detection_util');

load('stage_1_detection_result.mat')
annotation = annotation;
detection_result = detection_result;
step = step;

data_path = '/Users/jimmy/Desktop/images/bmvc17_soccer/DataSet_001/datasets/';

%% crop player and non-player image patches
N = numel(annotation);
train_player = [];
train_non_player = [];
validation_player = [];
validation_non_player = [];
patch_size = [32, 14];

train_num = int32(0.5 * N);
fp_tp_ratio = 3;
iou_threshold = 0.7;
% get training data
for i = [1:step:N]
    
    tp_bbox = annotation(i).bbox;
    fp_bbox = detection_result(i).stage_one;
    im_name = [data_path annotation(i).ImgName];
    
    im = imread(im_name);
    [positive, negative] = sample_from_false_positive(im, tp_bbox, fp_bbox, fp_tp_ratio, iou_threshold, patch_size);
    [negative2] = sample_negative_example(im, tp_bbox, patch_size, 3);   
    if i < train_num
        train_player = cat(4, train_player, positive);
        train_non_player = cat(4, train_non_player, negative);
        train_non_players = cat(4, train_non_player, negative2);
    else
        validation_player = cat(4, validation_player, positive);
        validation_non_player = cat(4, validation_non_player, negative);
        validation_non_player = cat(4, validation_non_player, negatvie2);
    end       
end

train_num = size(train_player, 4) + size(train_non_player, 4);
validation_num = size(validation_player, 4) + size(validation_non_player, 4);
num = train_num + validation_num;

% balance positive and negative exmples
meta.sets = {'train', 'val'};
meta.org_data = 'player, non-player image patches';
meta.image_mean = [123.8271, 108.3675, 82.0878]; % @todo

images.id = [1:num];
images.data = zeros(patch_size(1), patch_size(2), 3, num, 'single');  
images.label = zeros(1, num, 'double');
images.set = zeros(1, num, 'double');

% interleave player and non-player patches
index1 = [1:size(train_player, 4)];
index2 = [size(train_player, 4)+1:train_num];
index3 = [train_num+1:train_num + size(validation_player, 4)];
index4 = [train_num + size(validation_player, 4) + 1 : num]; 

%train_num = int32(0.8*num);
images.data(:,:,:,index1) = train_player;
images.data(:,:,:,index2) = train_non_player;
images.data(:,:,:,index3) = validation_player;
images.data(:,:,:,index4) = validation_non_player;
images.label(index1) = 2;
images.label(index2) = 1;
images.label(index3) = 2;
images.label(index4) = 1;
images.set([1:train_num]) = 1; % training
images.set([train_num+1:num]) = 2; % validation  2 for validation

% random shuffle 
index = randperm(num);
images.data = images.data(:,:,:,index);
images.label = images.label(index);
images.set = images.set(index);

% subtract mean value
m = [123.8271  108.3675   82.0878];
m = reshape(m, [1, 1, 3]);
im_mean = repmat(m, patch_size(1), patch_size(2), 1);
for i = [1:num]
    images.data(:,:,:,i) = images.data(:,:,:,i) - im_mean;
end

imdb.meta = meta;
imdb.images = images;













