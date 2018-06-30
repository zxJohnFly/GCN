function [A, Z, sigma] = AffinityMatrix(X, Anchor, s, sigma,Dis)
[n,~] = size(X);
m = size(Anchor,1);

%% get Z
Z = zeros(n,m);
if ~exist('Dis','var')
    Dis = EuDist2(X,Anchor,0);
end

clear X;
clear Anchor;

val = zeros(n,s);
pos = val;
for i = 1:s
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]';
    Dis(tep) = 1e60; 
end
clear Dis;
clear tep;

if sigma == 0
   sigma = mean(val(:,s).^0.5);
end

val = exp(-val/(1/1*sigma^2));
val = repmat(sum(val,2).^-1,1,s).*val; %% normalize
tep = (pos-1)*n+repmat((1:n)',1,s);
Z(tep) = val;
Z = sparse(Z);
A = 0;
%A = Z*(diag(1./sum(Z))*Z');
