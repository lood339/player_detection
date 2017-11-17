clear
close all

vl_setupnn;

f = 1/100;
net.layers = {};

lr = [.1 2] ; %based LR

% stage one
% block 1
net.layers{end+1} = struct('name', 'conv_m1', 'type', 'conv', ...
                           'weights', {{f*randn(3, 3, 3, 16, 'single'), zeros(1, 16, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0);

net.layers{end+1} = struct('type', 'relu', 'leak', 0);
net.layers{end+1} = struct('name', 'pool_m1', 'type', 'pool', ...
                           'method', 'max', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0);
                       
net.layers{end+1} = struct('name', 'pool_b1', 'type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5);        


% block 2                       
net.layers{end+1} = struct('name', 'conv_b1', 'type', 'conv', ...
                           'weights', {{f*randn(14, 6, 16, 2, 'single'), zeros(1, 2, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0);

net.layers{end+1} = struct('type', 'relu', 'leak', 0);

%net.layers{end+1} = struct('type', 'softmaxloss');

%{
% Meta parameters
net.meta.inputSize = [64 32 3] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 10 ;
net.meta.trainOpts.continue = true ;
net.meta.trainOpts.gpus = [] ;

% Fill in defaul values
net = vl_simplenn_tidy(net) ;

vl_simplenn_display(net);
im = randn(64, 32, 3, 'single');
res = vl_simplenn(net, im);
%}


