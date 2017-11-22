function [bbox_prob] = one_stage_player_detection(net, im, rgb_mean, scales, ... 
                                                  player_patch_size, prob_threshold, nms_threshold)
%function [bbox_prob] = one_stage_player_detection(net, im, rgb_mean, scales, ... 
%                                                  player_patch_size, prob_threshold, nms_threshold)                                              
% net: a classification network with dilation
% im: input image, three channel
% rgb_mean: mean value of image (three channel), [123.8271  108.3675
% 82.0878]
% scales: scales of the image, [1.0, 0.8, 0.5, 0.4, 0.3];
% player_patch_size: [h, w], [32, 14]
% prob_threshold: 0.8
% nms_threshold: non-maximum suppression threshold, 0.3
% output: bbox_prob, N * 5, [x, y, w, h, probability]
assert(length(size(im)) == 3);
assert(length(rgb_mean) == 3);
assert(length(player_patch_size) == 2);


net.layers{end} = struct('type', 'softmax');


player_patch_h = player_patch_size(1);
player_patch_w = player_patch_size(2);

player_threshold = prob_threshold;
rgb_mean = reshape(rgb_mean, [1, 1, 3]);

org_im = im;
%org_im = imread('0460.jpg');
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
    im_mean = repmat(rgb_mean, [im_h, im_w, 1]);
    
    
    im = single(im);
    im = im - single(im_mean);

    res = [];
    res = vl_simplenn(net, im, [], res, 'mode', 'test');    
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
%figure; imagesc(avg_prob); title('Average player probability'); axis equal;

bbox_prob = nms(player_bbox_prob, nms_threshold);

end