clear
close all

load('net_1_raw.mat')
load('imdb_v1.mat')

net.meta.trainOpts.batchSize = 20;
net.meta.trainOpts.numEpochs = 10;
% Train network
[net, info] = cnn_train(net, imdb, @getBatch, net.meta.trainOpts);
