
clear;
clc;
close all;

%%
% 生成GLCM分支的训练样本
PatSize = 7;
SamSize = 14;

neg_idx = load('neg_idx.mat');
pos_idx = load('pos_idx');
tst_idx = load('tst_idx');
neg_idx = neg_idx.neg_idx;
pos_idx = pos_idx.pos_idx;
tst_idx = tst_idx.tst_idx;

%%

tic

im1 = load('Guangzhou_3DGLCM12_patch13_Level7.mat');
im2 = load('Guangzhou_3DGLCM21_patch13_Level7.mat');
im1 = im1.GLCM12_result;
im2 = im2.GLCM21_result;

im1 = imresize(im1, [SamSize/2, SamSize], 'bilinear');
im2 = imresize(im2, [SamSize/2, SamSize], 'bilinear');

im=cat(1,im1,im2);

%%
sam_num = 6000;  
pos_num = 3000;

if length(neg_idx) < 3000  
    neg_num = length(neg_idx);
else
    neg_num = sam_num - pos_num;
end

% --- prepare the traninng data ---
trn_pat = zeros(SamSize, SamSize, 2*sam_num);
trn_lab = zeros(2, 2*sam_num);
trn_lab(1, 1:2*neg_num) = 1; 
trn_lab(2, 1+2*neg_num:2*sam_num) = 1; 


for i = 1:neg_num
    trn_pat(:,:,i) = im(:,:, neg_idx(i));
    
    idx1 = ceil(neg_num*rand()); 
    idx2 = ceil(neg_num*rand());
    ratio = rand();
    trn_pat(:,:,neg_num+i) = im(:,:,neg_idx(idx1))*ratio + ...
        im(:,:,neg_idx(idx2))*(1-ratio);
end

for i = 1 : pos_num %生成同等数量的虚拟正样本（不变样本）
    trn_pat(:,:, i+2*neg_num) = im(:,:, pos_idx(i));

    idx1 = ceil(pos_num*rand()); 
    idx2 = ceil(pos_num*rand());
    ratio = rand();
    trn_pat(:,:, i+2*neg_num+pos_num) = im(:,:,pos_idx(idx1))*ratio + ...
        im(:,:,pos_idx(idx2))*(1-ratio);
end
clear idx1 idx2 i ratio;

tst_pat = zeros(SamSize, SamSize,numel(tst_idx));
for i = 1 : numel(tst_idx)
  tst_pat(:,:,i) = im(:,:, tst_idx(i));
end

toc

clear CM_mid Data im;

glcm_trn = trn_pat;
glcm_tst = tst_pat;
save('Guangzhou_3DGLCM_non_trn.mat','glcm_trn','-v7.3');  
save('Guangzhou_3DGLCM_non_tst.mat','glcm_tst','-v7.3');
