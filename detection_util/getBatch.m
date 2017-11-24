function [im, labels] = getBatch(imdb, batch)
im = imdb.images.data(:,:,:,batch);
im = reshape(im, 32, 14, 3, []);
labels = imdb.images.label(1, batch);
end