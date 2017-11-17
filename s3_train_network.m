clear
close all

load('net_raw_1.mat')
load('imdb_v1.mat')

net.meta.trainOpts.batchSize = 20;
net.meta.trainOpts.numEpochs = 20;
% Train network
[net, info] = cnn_train(net, imdb, @getBatch, net.meta.trainOpts);
