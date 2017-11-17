clear
close all

load('net_1.mat')


net.layers{end} = struct('type', 'softmax', 'name', 'softmax');

N = numel(net.layers);
for i = [1:N]
    l = net.layers{i};
    switch l.name
        case 'conv_m1'
        
        case 'pool_m1'
            l.type = 'dpool';
            l.dilate = 2;
            l.stride = 1;
        case 'pool_b1'
            l.type = 'dpool';
            l.dilate = 4;
            l.stride = 1;
        case 'conv_b1'
            l.dilate = 4;
    end
    net.layers{i} = l;    
end






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


