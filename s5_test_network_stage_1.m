clear all
close all

% test detection using dilated network
load('./network/net_stage_1_dilation.mat')

net.layers{end} = struct('type', 'softmax');


player_patch_h = 32;
player_patch_w = 14;
im_h = 360;
in_w = 640;
threshold = 0.9;  % as a player

im = imread('./demo_image/0460.jpg');
im = imresize(im, [im_h, in_w]);

load('./dataset/im_mean.mat');
m = m;
m = reshape(m, [1, 1, 3]);

im_mean = repmat(m, [im_h, in_w, 1]);


org_im = im;
im = single(im);
im = im - single(im_mean);

res = [];
res = vl_simplenn(net, im, [], res, 'mode', 'test');


prob = res(end).x(:,:,2);

figure; imagesc(prob);
axis equal;




[rows, cols] = find(prob > threshold);


figure;
imshow(org_im);
hold on;
for i = [1:length(rows)]
    y = rows(i);
    x = cols(i);
    x = max(1, x);
    y = max(1, y);
    rectangle('Position', [x, y, player_patch_w, player_patch_h], 'EdgeColor', 'red');      
end




