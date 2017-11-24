function [positive, negative] = sample_from_false_positive(image, positive_bbox, false_positive_bbox, ... 
                                fp_tp_ratio, iou_threshold, patch_size)
% image: original image                           
% positive_bbox: bounding box of positive example
% false_positive_bbox: bounding box of false positive, N * 5, last colum is
% probability, candidate as false positive example
% fp_tp_ratio: false positive vs true positive ratio: 3
% iou_threshold: false positive IoU threshold: 0.7, lower than the
% threshold ad false positive
% patch_size: resized image size [h, w]

assert(length(image) >= 2);
assert(size(positive_bbox, 2) >= 4);
assert(size(false_positive_bbox, 2) >= 5);
assert(length(patch_size) == 2);

patch_h = patch_size(1);
patch_w = patch_size(2);

tp_bbox = int32(positive_bbox(:, [1:4])); % true positive
fp_bbox = int32(false_positive_bbox);
im = image;
[h, w, c] = size(im);
[N, d] = size(tp_bbox);
fp_num = int32(N * fp_tp_ratio);  % false positive number
assert(fp_num >= N);
assert(d >= 4);

positive = zeros(patch_h, patch_w, c, N);
negative = zeros(patch_h, patch_w, c, fp_num);

% crop positive area
for i = [1:N]
    patch = imcrop(im, tp_bbox(i, [1:4]));
    %figure; imshow(patch); title('Positive example'); pause(0.5);
    patch = imresize(patch, [patch_h, patch_w]);
    positive(:,:,:,i) = patch;    
end

% crop negative examples
valid_index = 1;
for i = [1:size(fp_bbox, 1)]
    max_iou = 0;    
    cur_bbox = fp_bbox(i, [1:4]);
    for j = [1:N]
        iou = bboxOverlapRatio(cur_bbox, tp_bbox(j, [1:4]));       
        if iou > max_iou
            max_iou = iou;            
        end
        if max_iou > iou_threshold
            break;  % true positive
        end
    end
    if max_iou > iou_threshold
        continue;  % true positive
    end
    
    patch = imcrop(im, cur_bbox);
    %figure; imshow(patch); title('Negative example'); pause(0.5);
    patch = imresize(patch, [patch_h, patch_w]);
    negative(:,:,:,valid_index) = patch;
    valid_index = valid_index + 1;
    if valid_index > fp_num
        break;
    end
end


if valid_index == 1
    negative = [];
else
    assert(valid_index > 1);
    negative = negative(:,:,:,[1:valid_index-1]);
end

end


function [x, y] = sample_bbox_in_image(h, w, patch_h, patch_w)
% randomly sample a point as the top left of the rectangle
x = randi(w);
y = randi(h);
if x > w - patch_w
    x = w - patch_w;
end
if y > h - patch_h
    y = h - patch_h;    
end

end