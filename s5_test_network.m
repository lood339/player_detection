clear
close all

%load('net_1.mat')
load('net_1_dilated.mat')
%load('imdb_v1.mat')

net.layers{end} = struct('type', 'softmax');


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
prob = res(end).x(:,:,2);

%{
scale = 1.0*size(im, 1)/size(prob, 1);
[row, col] = find(prob > threshold);
row = scale * row;
col = scale * col;

figure;
imshow(org_im);
hold on;
for i = [1:length(row)]
    y = row(i) - 32/2;
    x = col(i) - 64/2;
    x = max(1, x);
    y = max(1, y);
    rectangle('Position', [x, y, 32, 64], 'EdgeColor', 'red');      
end
%}




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


