
clear;
clc;
close all;

%%
% 生成image分支的训练样本

addpath('./Utils');

PatSize = 13;
SamSize = 26;

neg_idx = load('neg_idx.mat');
pos_idx = load('pos_idx');
tst_idx = load('tst_idx');
neg_idx = neg_idx.neg_idx;
pos_idx = pos_idx.pos_idx;
tst_idx = tst_idx.tst_idx;

Data=load('Guangzhou_filter.mat');
Data=Data.result;
im1=Data(:,:,1);im2=Data(:,:,2);
im1 = double(im1);
im2 = double(im2);
%%
%------------------------------------------------------------------------------------------------------------------------------------------------------------
tic

[ylen, xlen] = size(im1);

mag = (PatSize-1)/2;
imTmp = zeros(ylen+PatSize-1, xlen+PatSize-1);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im1; % imTmp为对im1边缘零填充
im1 = im2col_general(imTmp, [PatSize, PatSize]);
imTmp((mag+1):end-mag,(mag+1):end-mag) = im2; 
im2 = im2col_general(imTmp, [PatSize, PatSize]);
clear imTmp mag;


im = zeros(SamSize, SamSize, ylen*xlen);

parfor idx = 1 : size(im1, 2)
    imtmp1 = reshape(im1(:, idx), [PatSize, PatSize]);
    imtmp2 = reshape(im2(:, idx), [PatSize, PatSize]);
    
    imtmp1 = imresize(imtmp1, [SamSize/2, SamSize], 'bilinear'); 
    imtmp2 = imresize(imtmp2, [SamSize/2, SamSize], 'bilinear');
    
    im(:,:,idx) = [imtmp1; imtmp2];
end
clear im1 im2 idx imtmp1 imtmp2;

sam_num = 6000; 
pos_num = 3000;

if length(neg_idx) < 3000  
    neg_num = length(neg_idx);
else
    neg_num = sam_num - pos_num;
end

trn_pat = zeros(SamSize, SamSize, 2*sam_num);
trn_lab = zeros(2, 2*sam_num);
trn_lab(1, 1:2*neg_num) = 1;  
trn_lab(2, 1+2*neg_num:2*sam_num) = 1; 

label=trn_lab(1,:); 
label=label';

for i = 1:neg_num
    trn_pat(:,:,i) = im(:,:, neg_idx(i));
    
    idx1 = ceil(neg_num*rand());
    idx2 = ceil(neg_num*rand());
    ratio = rand();
    trn_pat(:,:,neg_num+i) = im(:,:,neg_idx(idx1))*ratio + ...
        im(:,:,neg_idx(idx2))*(1-ratio);
end

for i = 1 : pos_num 
    trn_pat(:,:, i+2*neg_num) = im(:,:, pos_idx(i));

    % virtual sample
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

clear CM_mid Data im; 

toc

image_trn = trn_pat;
image_tst = tst_pat;
save('Guangzhou_image_non_trn.mat','image_trn','-v7.3');
save('Guangzhou_image_non_tst.mat','image_tst','-v7.3');
save('Guangzhou_Pseudo_label.mat','label','-v7.3'); 
