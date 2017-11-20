close all;
clear all;


load('net_stage_2.mat')

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
            
        case 'conv_m2'
            l.dilate = 2;
            l.stride = 1;           
            'conv_m2'
            
         case 'pool_m2'
             l.type = 'dpool';            
            l.dilate = 4;
            l.stride = 1;            
            'pool_m2'
            
        case 'conv_b2'
            l.dilate = 4;
            l.stride = 1;
            'conv_b2'
    end
    net.layers{i} = l;    
end

%vl_simplenn_display(net);

im = imread('0460.jpg');
im = imresize(im, [360, 640]);

load('im_mean.mat');
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


x8 = res(8).x(:,:,:);
for i = [1:size(x8, 3)]
    %figure; imagesc(x8(:,:,i));
    temp_x = x8(:,:,i);
    sum(isnan(temp_x(:)));
end
%{
%figure; imagesc(x6);
x7 = res(7).x(:,:,:); %figure; imagesc(x7);
for i = [1:size(x7, 3)]
    %figure; imagesc(x7(:,:,i));
    temp_x = x7(:,:,i);
    sum(isnan(temp_x(:)))
end
%}

prob = res(end).x(:,:,2);

figure; imagesc(prob);
axis equal;






%{
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


