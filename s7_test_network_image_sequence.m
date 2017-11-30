clear all
close all

load('net_1_dilation.mat')
net.layers{end} = struct('type', 'softmax');

% detection parameters
player_patch_size = [32, 14];
scales = [1.0, 0.8, 0.5, 0.4, 0.3];
player_threshold = 0.85;
nms_threshold = 0.3;

% image means
load('im_mean.mat');
rgb_mean = m;

load('annotation_1.mat')
annotation = annot;

data_path = '/Users/jimmy/Desktop/images/bmvc17_soccer/DataSet_001/datasets/';

%% crop player and non-player image patches
N = numel(annotation);
step = 10;

detection_result = [];

for i = [1:step:N]
   
    im_name = [data_path annotation(i).ImgName];    
    im = imread(im_name);
    tic
    player_bbox = one_stage_player_detection(net, im, rgb_mean, scales, player_patch_size, player_threshold, nms_threshold); 
    toc
    detection_result(i).stage_one = player_bbox;
    
    size(player_bbox)
    figure(1);
    imshow(im);
    hold on;     
    for j = [1:length(player_bbox)]
        cur_bbox = player_bbox(j,[1:4]);
        rectangle('Position', cur_bbox, 'EdgeColor', 'green', 'LineWidth', 1);      
    end
    pause(0.1);
end

















