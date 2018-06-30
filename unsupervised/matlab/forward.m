addpath('/mnt/disk1/');
addpath('/mnt/disk1/ImageNet1M/ImageNettiny/');
addpath('~/HELPFUN/');

fprintf('loading.. \n');
load('ImageNett_model_32.mat');
load('adj_matrix_5.mat', 'A');
load('sample_anchor_graph.mat');
load('test_anchor_graph.mat', 'tZ');
load('ImageNet128K.mat', 'traindata', 'testdata', 'cateTrainTest');
load('anchor1000.mat');

traindata = double(traindata);
testdata = double(testdata);

tran1 = Z*(diag(1./sum(Z))*tZ');

w1 = double(w1);
w2 = double(w2);
w3 = double(w3);
b1 = double(b1);
b2 = double(b2);
b3 = double(b3);

A = sign(A);
A = sparse(bsxfun(@rdivide, A, sum(A, 1)));

adj = sign(tran1);
adj(isnan(adj)) = 0; 
adj1 = sparse(bsxfun(@rdivide, adj, sum(adj, 1)));
adj1(isnan(adj1)) = 0;
adj2 = sparse(bsxfun(@rdivide, adj', sum(adj', 1)));
adj2(isnan(adj2)) = 0;

%% forward graph convolutional network
act1 = tanh(adj1*(bsxfun(@plus, testdata*w1, b1)));
act2 = tanh(A*(bsxfun(@plus, act1*w2, b2)));
act3 = tanh(adj2*(bsxfun(@plus, act2*w3, b3)));

tB = sign(act3); 
B = [];

batches = mat2cell(traindata, diff(round(linspace(0, size(traindata, 1),10)))); 
for i = 1:length(batches)
    fprintf('batch %d / %d\n', i, length(batches));
    batch = batches{i};
    [~, tZ] = AffinityMatrix(batch, anchor, 5, 0);
    tadj = sign(Z*diag(1./sum(Z))*tZ');
    tadj(isnan(tadj)) = 0;

    adj1 = sparse(bsxfun(@rdivide, tadj, sum(tadj, 1)));
    adj1(isnan(adj1)) = 0;
    adj2 = sparse(bsxfun(@rdivide, tadj', sum(tadj', 1)));
    adj2(isnan(adj2)) = 0; 

    act1 = tanh(adj1*(bsxfun(@plus, batch*w1, b1)));
    act2 = tanh(A*(bsxfun(@plus, act1*w2, b2)));
    act3 = tanh(adj2*(bsxfun(@plus, act2*w3, b3)));

    B = [B; sign(act3)];     
end

hammTrainTest = 0.5*(32 - B*tB');
[~, HammingRank] = sort(hammTrainTest, 1);
cat_apcal_simply(cateTrainTest, HammingRank)
cat_ap_topK(cateTrainTest,HammingRank, 500)
