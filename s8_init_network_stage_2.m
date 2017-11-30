clear all
close all

% initialize network (stage 2) from stage 1
% conv_m1 and pool_m1 are shared by stage-1 network and stage-2 network
% input: stage 1 network (un-dilated)
% output: initial stage 2 network
% note: conv_m1 is not updated in stage 2 training

addpath('./network');

org_net = load('net_stage_1.mat');
org_layers = org_net.net.layers;

f = 1/100;
net.layers = {};

for i = 1:length(org_layers)
    l = org_layers{i};
    switch l.name
        case 'conv_m1'
            l.learningRate = [0, 0]; % do not update weights
            l.weightDecay = [0, 0];
        case 'pool_m1'
            % remove layers after this layer        
    end
    net.layers{end+1} = l;
    if strcmp(l.name, 'pool_m1')
        break;
    end
end

clear org_net
% remove the layer after pool_m1


lr = [.1 2] ; %based LR
% stage 2                       
net.layers{end+1} = struct('name', 'conv_m2', 'type', 'conv', ...
                           'weights', {{f*randn(3, 3, 16, 16, 'single'), zeros(1, 16, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0);
net.layers{end+1} = struct('type', 'relu', 'leak', 0);

net.layers{end+1} = struct('name', 'pool_m2', 'type', 'pool', ...
                           'method', 'avg', ...
                           'pool', [3 3], ...
                           'stride', 2, ...
                           'pad', 0);
                     
net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5); 

% fc                    
net.layers{end+1} = struct('name', 'conv_b2', 'type', 'conv', ...
                           'weights', {{f*randn(5, 1, 16, 2, 'single'), zeros(1, 2, 'single')}}, ...
                           'learningRate', lr, ...
                           'stride', 1, ...
                           'pad', 0);

net.layers{end+1} = struct('type', 'relu', 'leak', 0);

net.layers{end+1} = struct('type', 'softmaxloss');


% Meta parameters
net.meta.inputSize = [32 14 3] ;
net.meta.trainOpts.learningRate = 0.001 ;
net.meta.trainOpts.numEpochs = 20 ;
net.meta.trainOpts.batchSize = 10 ;
net.meta.trainOpts.continue = true ;
net.meta.trainOpts.gpus = [] ;

net = vl_simplenn_tidy(net) ;

vl_simplenn_display(net);
im = randn(32, 14, 3, 'single');
%res = vl_simplenn(net, im);

