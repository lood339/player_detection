clear all
close all

load('net_1_dilation.mat')

net.layers{end} = struct('type', 'softmax');


player_patch_h = 32;
player_patch_w = 14;
im_h = 360;
in_w = 640;

im = imread('0460.jpg');
im = imresize(im, [im_h, in_w]);

load('im_mean.mat');
m = m;
m = reshape(m, [1, 1, 3]);

im_mean = repmat(m, [im_h, in_w, 1]);


org_im = im;
im = single(im);
im = im - single(im_mean);

res = [];
res = vl_simplenn(net, im, [], res, 'mode', 'test');

threshold = 0.9;
prob = res(end).x(:,:,2);

figure; imagesc(prob);


%scale = 1.0*size(im, 1)/size(prob, 1);
%prob = resizem(prob, [360, 640]);
dy = int32((size(im, 1) - size(prob, 1))/2);
dx = int32((size(im, 2) - size(prob, 2))/2);

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


