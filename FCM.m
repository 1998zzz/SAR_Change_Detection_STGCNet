
clear;
clc;
close all;

%%
addpath('./Utils');

fprintf(' ... ... read image file ... ... ... ....\n');
Data=load('Guangzhou.mat');
Data=Data.result;
im1=Data(:,:,1);im2=Data(:,:,2);

fprintf(' ... ... read image file finished !!! !!!\n\n');

im1 = double(im1);
im2 = double(im2);
[ylen, xlen] = size(im1);
im_di = di_gen(im1, im2);
pixel_vector = reshape(im_di, ylen*xlen, 1);

fprintf('... ... clustering begin ... ...\n');
im_lab = hclustering(pixel_vector, im_di);
fprintf('@@@ @@@ clustering finished @@@@\n');

clear pixel_vector im_di;

select_size=25;
CM_mid=padarray(im_lab,[floor(select_size/2),floor(select_size/2)]);
tic
    for y=1:ylen
        for x=1:xlen
            sample=CM_mid(x:(x+select_size-1),y:(y+select_size-1));
            pixels=find(sample==sample(1+floor(select_size/2),1+floor(select_size/2)));
            if(length(pixels)/(select_size*select_size)<0.55)
                im_lab(x,y)=0.5;
            end
        end
    end
toc

pos_idx = find(im_lab == 1);   
neg_idx = find(im_lab == 0);   
tst_idx = find(im_lab == 0.5); 


rand('seed', 2);                            
pos_idx = pos_idx(randperm(numel(pos_idx))); 
neg_idx = neg_idx(randperm(numel(neg_idx)));

%%
save('pos_idx.mat','pos_idx','-v7.3');
save('neg_idx.mat','neg_idx','-v7.3');
save('tst_idx.mat','tst_idx','-v7.3');
save('Guangzhou_Pseudo_imlab.mat','im_lab','-v7.3'); 