function [negative] = sample_negative_example(image, bounding_box, patch_size,...
    ratio)
% [negative] = sample_negative_example(bounding_box, image, patch_size)
% bounding_box: bounding box of positive examples
% image: original imae
% patch_size: [h,w] of resized the image patch
% ratio: negative vs positive ratio
% assume: non-positive area has negative examples
assert(size(bounding_box, 2) >= 4);
assert(length(image) >= 2);
assert(length(patch_size) == 2);

patch_h = patch_size(1);
patch_w = patch_size(2);

bbox = int32(bounding_box);
im = image;
[h, w, c] = size(im);
[N, d] = size(bbox);
assert(d >= 4);

M = int32(N * ratio);
negative = zeros(patch_h, patch_w, c, M);


% randomly sample negative examples
% overlapRatio = bboxOverlapRatio(bboxA,bboxB)
iou_threshold = 0.3;
mask = zeros(h, w);
for i = [1:N]
    x = bbox(i, 1);
    y = bbox(i, 2);
    x_w = bbox(i, 3);
    y_h = bbox(i, 4);
    mask([y:y+y_h], [x: x+x_w]) = 1.0;
end
%imagesc(mask);

valid_index = 1;
for i = [1:M]
    [x, y] = sample_bbox_in_image(h, w, patch_h, patch_w);
    sub_mask = mask([y:y+patch_h], [x:x+patch_w]);
    if(sum(sub_mask(:))/(patch_w*patch_h) < iou_threshold)
        patch = imcrop(im, [x, y, patch_w-1, patch_h-1]);
        negative(:,:,:,valid_index) = patch;
        valid_index = valid_index + 1;
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


