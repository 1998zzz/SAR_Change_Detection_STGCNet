clear;
clc;
close all

tic
%%
%原始数据加载
Data=load('Guangzhou_filter.mat');
Data=Data.result;

%%
[row,col,chs]=size(Data);

level=7;
Data(Data>512)=512;
Data(Data<2)=2;
Data=log2(Data/2);
Data=Data./8*level;
Data(Data>(level-1))=(level-1);
Data = Data + 1;

% 滑动窗口取patch
patch_size=13;
Data_mid=padarray(Data,[floor(patch_size/2),floor(patch_size/2)],'replicate');

% 计算GLCM
counter=1;
dx=1;dy=1;dz=1;
GLCM12_result=zeros(level,level,row*col);
GLCM21_result=zeros(level,level,row*col);
%注意一下循环时x、y、z的顺序
    for y=1:col
        for x=1:row
            patch=floor(Data_mid(x:(x+patch_size-1),y:(y+patch_size-1),:));
            [glcm_81_T12]=glcm1(patch(:,:,1),patch(:,:,1),dx,dy,dz,level);
            [glcm_81_T21]=glcm1(patch(:,:,2),patch(:,:,2),dx,dy,dz,level);
            glcm_sum1_T12=sum(glcm_81_T12,3);
            glcm_sum1_T21=sum(glcm_81_T21,3);
            [glcm_82_T12]=glcm2(patch(:,:,1),patch(:,:,2),dx,dy,dz,level);
            [glcm_82_T21]=glcm2(patch(:,:,2),patch(:,:,1),dx,dy,dz,level);
            glcm_sum2_T12=sum(glcm_82_T12,3);
            glcm_sum2_T21=sum(glcm_82_T21,3);
            glcm_sum_T12=glcm_sum1_T12+glcm_sum2_T12; % All
            glcm_sum_T21=glcm_sum1_T21+glcm_sum2_T21; % All
            GLCM12_result(:,:,counter)=glcm_sum_T12;
            GLCM21_result(:,:,counter)=glcm_sum_T21;
            counter=counter+1;
        end
    end   

toc

%%
save('Guangzhou_3DGLCM12_patch13_Level7_pad0.mat','GLCM12_result','-v7.3');
save('Guangzhou_3DGLCM21_patch13_Level7_pad0.mat','GLCM21_result','-v7.3');
%%
% GLCM函数
function [glcm_81]=glcm1(T1,T2,dx,dy,dz,level)

 [W,L,C]=size(T1);
    glcm_81=zeros(level,level,8);
    % 方向1
    x2range = mod((1:W) + 0 - 1, W) + 1; 
    y2range = mod((1:L) + dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,1) = reshape(Value, [level, level]);
    % 方向2
    x2range = mod((1:W) + dx - 1, W) + 1;
    y2range = mod((1:L) + dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level],  ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
     glcm_81(:,:,2) = reshape(Value, [level, level]);
    % 方向3
    x2range = mod((1:W) + dx - 1, W) + 1;
    y2range = mod((1:L) + 0 - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,3) = reshape(Value, [level, level]);
    % 方向4
    x2range = mod((1:W) + dx - 1, W) + 1;
    y2range = mod((1:L) - dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,4) = reshape(Value, [level, level]);
    % 方向5
    x2range = mod((1:W) + 0 - 1, W) + 1;
    y2range = mod((1:L) - dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,5) = reshape(Value, [level, level]);
     % 方向6
    x2range = mod((1:W) - dx - 1, W) + 1;
    y2range = mod((1:L) - dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,6) = reshape(Value, [level, level]);   
    % 方向7
    x2range = mod((1:W) - dx - 1, W) + 1;
    y2range = mod((1:L) + 0 - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,7) = reshape(Value, [level, level]);  
    % 方向8
    x2range = mod((1:W) - dx - 1, W) + 1;
    y2range = mod((1:L) + dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_81(:,:,8) = reshape(Value, [level, level]); 
end

function [glcm_82]=glcm2(T1,T2,dx,dy,dz,level)

 [W,L,C]=size(T1);
    glcm_82=zeros(level,level,8);
    % 方向1
    x2range = mod((1:W) + 0 - 1, W) + 1; 
    y2range = mod((1:L) + dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,1) = reshape(Value, [level, level]);
    % 方向2
    x2range = mod((1:W) + dx - 1, W) + 1;
    y2range = mod((1:L) + dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level],  ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
     glcm_82(:,:,2) = reshape(Value, [level, level]);
    % 方向3
    x2range = mod((1:W) + dx - 1, W) + 1;
    y2range = mod((1:L) + 0 - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,3) = reshape(Value, [level, level]);
    % 方向4
    x2range = mod((1:W) + dx - 1, W) + 1;
    y2range = mod((1:L) - dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,4) = reshape(Value, [level, level]);
    % 方向5
    x2range = mod((1:W) + 0 - 1, W) + 1;
    y2range = mod((1:L) - dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,5) = reshape(Value, [level, level]);
     % 方向6
    x2range = mod((1:W) - dx - 1, W) + 1;
    y2range = mod((1:L) - dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,6) = reshape(Value, [level, level]);   
    % 方向7
    x2range = mod((1:W) - dx - 1, W) + 1;
    y2range = mod((1:L) + 0 - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,7) = reshape(Value, [level, level]);  
    % 方向8
    x2range = mod((1:W) - dx - 1, W) + 1;
    y2range = mod((1:L) + dy - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,8) = reshape(Value, [level, level]); 
    % 方向9
    x2range = mod((1:W) + 0 - 1, W) + 1;
    y2range = mod((1:L) + 0 - 1, L) + 1;
    T_new = T2(x2range, y2range);
    xlist = T_new(:);
    ylist = T1(:);
    indlist = sub2ind([level, level], ylist, xlist);
    [Value    Count] = hist(indlist, 1:level*level);
    glcm_82(:,:,9) = reshape(Value, [level, level]); 
end
