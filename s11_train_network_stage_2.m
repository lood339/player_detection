clear
close all

% train classify network
% input: raw network and training data
addpath('./network_raw');

load('net_stage_2_init_from_net_1.mat')
load('imdb_v2_stage_2.mat')

net.meta.trainOpts.batchSize = 20;
net.meta.trainOpts.numEpochs = 20;
% Train network
[net, info] = cnn_train(net, imdb, @getBatch, net.meta.trainOpts);
