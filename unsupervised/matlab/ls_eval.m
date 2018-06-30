addpath('~/HELPFUN');
addpath('/mnt/disk1/ImageNet1M');
addpath('/mnt/disk1/ImageNet1M/ImageNettiny');

%load('sample_anchor_graph.mat')
%sZ = Z;
load('mean.mat');
%load('anchor1000.mat');
load('ILSVRC2012_caffe_CNN.mat')

%nbits = 128;
%load([num2str(nbits) '.mat'])
%sB = Z;

traindata = bsxfun(@minus, traindata, m);
testdata = bsxfun(@minus, testdata, m);

%[~, Z] = AffinityMatrix(traindata, anchor, 5, sigma);
%clear traindata
%[~, tZ] = AffinityMatrix(testdata, anchor, 5, sigma);
%clear testdata

%sB = sZ'*sB;
%tB = single(sign((tZ*dITQiag(1./sum(tZ)))*sB)); 
%B = single(sign((Z*diag(1./sum(Z)))*sB));
%
%clear sB sZ Z tZ;

load('~/Hashing/ITQ/PCA_ITQ_ImageNet128K.mat', 'HashFunc');

num_test = size(testdata, 1);
num_block = ceil(num_test/1e3);
tem_cateMAP = zeros(num_block, 1);
tem_cateTop = zeros(num_block, 1);

tem_pre = zeros(num_block, length(traingnd));
tem_rec = zeros(num_block, length(traingnd));

W = HashFunc{1};
B = sign(traindata*W);
tB = sign(testdata*W);

clear traindata, testdata; 
tem_testgnd = cell(num_block, 1);
tem_tB = cell(num_block, 1);

for block = 1:num_block
    if block ~= num_block
        ixxxx = (block-1)*1e3+1:block*1e3;
    else
        ixxxx = (block-1)*1e3+1:length(testgnd);
    end
    tem_testgnd{block} = testgnd(ixxxx);
    tem_tB{block} = tB(ixxxx, :);
end

if ~matlabpool('size')
    matlabpool 5;
end

parfor block = 1:num_block
    block
    hammTrainTest = 0.5*(16 - B*tem_tB{block}');

    [~, HammingRank] = sort(hammTrainTest, 1);
    cateTrainTest = bsxfun(@eq, traingnd, tem_testgnd{block}');
    tem_cateMAP(block) = cat_apcal(traingnd, tem_testgnd{block}, HammingRank);
    tem_cateTop(block) = cat_ap_topK(cateTrainTest, HammingRank, 5000); 
    [tem_pre(block,:), tem_rec(block,:)] = evaluate_HammingRanking_category(traingnd, tem_testgnd{block}, HammingRank);
end

MAP = mean(tem_cateMAP, 0)
cateTop = mean(tem_cateTop)
pre = mean(tem_pre);
rec = mean(tem_rec);

save(['ITQ', num2str(nbits)], 'MAP', 'pre' , 'rec');
