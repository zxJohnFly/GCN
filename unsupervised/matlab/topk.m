addpath ~/HELPFUN

load('/mnt/disk1/NUSWIDE_concept.mat', 'cateTrainTest');
load('NUSWIDE_16.mat');

h = 0.5*(16-B*tB');
[~, HammingRank] = sort(h, 1);
[prec, rec] = evaluate_HammingRanking_category_multiclass(double(cateTrainTest), HammingRank);

save('Ours_NUSWIDE_16', 'rec', 'prec')
