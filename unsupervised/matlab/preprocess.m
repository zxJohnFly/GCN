addpath('~/HELPFUN');

dir = '/mnt/disk1/NUS_WIDE/';

load('/mnt/disk1/NUSWIDE_concept.mat');
n_train = 60000;

traindata = normalize(traindata);
testdata = normalize(testdata);

[~, anchor] = litekmeans(traindata, 1000, 'Maxiter', 15);
indc = randperm(size(traindata, 1), n_train);
sample = traindata(indc, :);

save([dir, 'anchor1000.mat'], 'anchor');
[~, Z, sigma] = AffinityMatrix(sample, anchor, 5, 0);
A = sparse(Z*(diag(1./sum(Z))*Z'));
save([dir, 'adj_matrix_5'], 'A');
save([dir, 'sample_anchor_graph'], 'Z', 'sigma');

[~, Z] = AffinityMatrix(traindata, anchor, 5, sigma);
[~, tZ] = AffinityMatrix(testdata, anchor, 5, sigma);
save([dir, 'anchor_graph'], 'Z');
save([dir, 'test_anchor_graph'], 'tZ');

save([dir, 'nus_wide'], 'cateTrainTest', 'sample');
