clear all
close all

org_net = load('./network/net_1_dilation.mat');
net1 = org_net.net;
clear org_net;

net1.layers{end} = struct('type', 'softmax');

org_net = load('./network/net_stage_2.mat');
net2 = org_net.net;
clear org_net;
net2.layers{end} = struct('type', 'softmax');


player_patch_h = 32;
player_patch_w = 14;

scales = [1.0, 0.8, 0.6, 0.4, 0.3];
player_threshold = 0.85;

% image means
load('./dataset/im_mean.mat');
m = m;
m = reshape(m, [1, 1, 3]);

org_im = imread('0460.jpg');
avg_prob = zeros(size(org_im, 1), size(org_im, 2));
player_bbox_prob = []; % top-left x, y, width, height, probability
for s_index = [1:length(scales)]
    s = scales(s_index);
    im = org_im;
    im_size = [size(im, 1), size(im, 2)];
    im_h = int32(im_size(1) * s);
    im_w = int32(im_size(2) * s);
    
    % resize image to a scaled version
    im = imresize(im, [im_h, im_w]);    
    im_mean = repmat(m, [im_h, im_w, 1]);
    
    
    im = single(im);
    im = im - single(im_mean);

    res = [];
    res = vl_simplenn(net1, im, [], res, 'mode', 'test');    
    cur_prob = res(end).x(:,:,2);
    
        
    cur_prob_map = zeros(im_h, im_w);
    cur_prob_map([1: size(cur_prob,1)],[1: size(cur_prob, 2)]) = cur_prob;
    
    %figure; imagesc(cur_prob_map); axis equal; 
    [cur_rows, cur_cols] = find(cur_prob_map > player_threshold);
    cur_location_size = zeros(length(cur_rows), 5);
    for j = [1:length(cur_rows)]
        cur_location_size(j, 1) = round(cur_cols(j)/s);
        cur_location_size(j, 2) = round(cur_rows(j)/s);
        cur_location_size(j, 3) = round(player_patch_w/s);
        cur_location_size(j, 4) = round(player_patch_h/s);
        cur_location_size(j, 5) = cur_prob_map(cur_rows(j), cur_cols(j));
    end
    player_bbox_prob = [player_bbox_prob; cur_location_size];    
    
    % resize probability map to the original image size
    avg_prob = avg_prob + resizem(cur_prob_map, [size(org_im, 1), size(org_im, 2)]);    
end

avg_prob = avg_prob ./length(scales);
figure; imagesc(avg_prob); title('Average player probability'); axis equal;

%% filt false positive by second stage
for i = [1:size(player_bbox_prob, 1)]
    patch = imcrop(org_im, player_bbox_prob(i, [1:4]));
    patch = imresize(patch, [player_patch_h, player_patch_w]);
    
    im_mean = repmat(m, [player_patch_h, player_patch_w, 1]);
    patch = single(patch) - single(im_mean);
    
    res = [];
    res = vl_simplenn(net2, patch, [], res, 'mode', 'test');    
    cur_prob = res(end).x(:,:,2);
    player_bbox_prob(i, 5) = cur_prob; 
end

player_threahold_2 = 0.7;
index = find(player_bbox_prob(:,5) > player_threahold_2);
player_bbox_prob = player_bbox_prob(index,:);


selected_bbox = nms(player_bbox_prob, 0.3);

figure;
imshow(org_im);
hold on;
for i = [1:size(selected_bbox, 1)]
    cur_bbox = selected_bbox(i,[1:4]);
    rectangle('Position', cur_bbox, 'EdgeColor', 'green', 'LineWidth', 1);      
end


