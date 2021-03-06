close all;
clear all;

% input: trained stage 1 network
% output: dilated staget 1 network
% purpose: add dilation after each pooling layer so that the network
%          can be applied to the whole imae

load('./network/net_stage_1.mat')

for i = 1:length(net.layers)
    l = net.layers{i};
    switch l.type
        case 'conv'
        w1 = l.weights{1};
        w2 = l.weights{2};
        sum(isnan(w1(:))) 
        sum(isnan(w2(:)))
    end
end

net.layers{end} = struct('type', 'softmax', 'name', 'softmax');

N = numel(net.layers);
for i = [1:N]
    l = net.layers{i};
    switch l.name
        case 'conv_m1'
        
        case 'pool_m1'
            l.type = 'dpool';
            l.dilate = 1;
            l.stride = 1;
            'pool_m1'
            
        case 'pool_b1'
            l.type = 'dpool';
            l.dilate = 2;
            l.stride = 1;
            'pool_b1'
            
        case 'conv_b1'
            l.dilate = 4;
            l.stride = 1;
            l.pad = 0;
            'conv_b1'
    end
    net.layers{i} = l;    
end

%vl_simplenn_display(net);

im = imread('./demo_image/0460.jpg');
im = imresize(im, [360, 640]);

load('./dataset/im_mean.mat');
m = m;
m = reshape(m, [1, 1, 3]);

im_mean = repmat(m, [360, 640, 1]);  

org_im = im;
im = single(im);
im = im - single(im_mean);

res = [];
res = vl_simplenn(net, im, [], res, 'mode', 'test');

threshold = 0.9;
size(res(end).x)

x6 = res(6).x(:,:,:);
for i = [1:size(x6, 3)]
    %figure; imagesc(x6(:,:,i));
    temp_x = x6(:,:,i);
    sum(isnan(temp_x(:)));
end
%figure; imagesc(x6);
x7 = res(7).x(:,:,:); %figure; imagesc(x7);
for i = [1:size(x7, 3)]
    %figure; imagesc(x7(:,:,i));
    temp_x = x7(:,:,i);
    sum(isnan(temp_x(:)))
end

prob = res(end).x(:,:,2);

figure; imagesc(prob);
axis equal;









