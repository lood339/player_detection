clear all
close all

load('net_1_dilation.mat')

net.layers{end} = struct('type', 'softmax');


player_patch_h = 32;
player_patch_w = 14;

scales = [1.0, 0.8, 0.5, 0.4, 0.3];
player_threshold = 0.85;

% image means
load('im_mean.mat');
m = m;
%m = reshape(m, [1, 1, 3]);

org_im = imread('0460.jpg');



%player_bbox = one_stage_player_detection(net, org_im, m, scales, [32, 14], 0.85, 0.3); 
figure;
imshow(org_im);
hold on;
for i = [1:length(player_bbox)]
    cur_bbox = player_bbox(i,[1:4]);
    rectangle('Position', cur_bbox, 'EdgeColor', 'green', 'LineWidth', 1);      
end











%scale = 1.0*size(im, 1)/size(prob, 1);
%prob = resizem(prob, [360, 640]);
%{

[rows, cols] = find(prob > threshold);
for i = [1:length(rows)]
    rows(i) = rows(i) + dy;
    cols(i) = cols(i) + dx;
end

figure;
imshow(org_im);
hold on;
for i = [1:length(rows)]
    y = rows(i) - player_patch_h/2;
    x = cols(i) - player_patch_w/2;
    x = max(1, x);
    y = max(1, y);
    rectangle('Position', [x, y, player_patch_w, player_patch_h], 'EdgeColor', 'red');      
end
%}





%{
% test on image patches
net.layers{end} = struct('type', 'softmax');

images = imdb.images;
N = length(images.label);
ground_truth = images.label;
prediction = [];

for i = [1:N]   
    res = [];
    im = images.data(:,:,:,i);
    res = vl_simplenn(net, im, [], res, 'mode', 'test');
    prob = res(end).x(:);
    [v, idx] = max(prob);
    prediction(i) = idx;    
end

c = confusionmat(ground_truth, prediction);
c/(sum(c(:)))
%}


