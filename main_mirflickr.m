clc;
clear;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end
dataset_name = {'mirflickr25k'};

%% load dataset
 for db = 1:length(dataset_name)
 dataset = dataset_name{db};
 load(['./datasets/',dataset,'.mat']);
%   rng('default');     
% make the training/test data zero-mean
id1 = all(I_tr==0,2);
id2 = all(T_tr==0,2);
L_tr1 = L_tr;
L_tr1(id1,:) = [];
L_tr2 = L_tr;
L_tr2(id2,:) = [];
I_tr (id1,:) = [];
T_tr (id2,:) = [];
%    R = randperm(size(L_tr1,1));
%    L_tr1 = L_tr1(R(1:end),:);
%    I_tr = I_tr(R(1:end),:);
%    L_tr2 = L_tr2(R(1:1500),:);
%    T_tr = T_tr(R(1:1500),:);
% I_te = bsxfun(@minus, I_te, mean(I_tr, 1));     
% I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));    
% T_te = bsxfun(@minus, T_te, mean(T_tr, 1));    
% T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));        
%% parameter setting
 nbits = [128];       maxItr = [10];              lambda = [10000];
 muta = [1];          theta = [0.001];       M = [0];
%  maxItr = 13;         
turn = 10;             func = 'linear';
l = 1; %excel writing parameter
%           maxItr = [1:50];             
%                  lambda = [0 0.1,1,10,100, 1000, 10000,100000, 1000000,10000000];
%                 muta =  [0.0001,0.001,0.01,0.1,1,10,100,1000,10000];
%                  theta = [0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100];
 %% start
  tic;
for bi = 1:length(nbits)
    for i = 1:length(maxItr)
        for j=1:length(lambda)
             for k = 1:length(muta)
               for r=1:length(theta)               
                     for v = 1:turn   
                      %% kernelization
                   fprintf('kernelizing...\n');
                   param.chunksize = 1000;
                   param.xchunks = floor(size(L_tr1,1)/param.chunksize);
                   param.ychunks = floor(size(L_tr2,1)/param.chunksize);
                   BMCHparam.nAnchors = 2000; 
                   [XKTrain,XKTest] = Kernelize(I_tr,I_te,BMCHparam.nAnchors); 
                   [YKTrain,YKTest] = Kernelize(T_tr,T_te,BMCHparam.nAnchors);
                   XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1));     
                   XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));    
                   YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1));     
                   YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));  

                   XChunk = cell(param.xchunks,1);
                   YChunk = cell(param.ychunks,1);
                   LXChunk = cell(param.xchunks,1);
                   LYChunk = cell(param.ychunks,1);
                   for subx = 1:param.xchunks-1
                        XChunk{subx,1} = XKTrain(param.chunksize*(subx-1)+1:param.chunksize*subx,:);
                        LXChunk{subx,1} = L_tr1(param.chunksize*(subx-1)+1:param.chunksize*subx,:);
                    end
                    for suby = 1:param.ychunks-1
                        YChunk{suby,1} = YKTrain(param.chunksize*(suby-1)+1:param.chunksize*suby,:);
                        LYChunk{suby,1} = L_tr2(param.chunksize*(suby-1)+1:param.chunksize*suby,:);
                    end
                    XChunk{param.xchunks,1} = XKTrain(param.chunksize*subx+1:end,:);
                    YChunk{param.ychunks,1} = YKTrain(param.chunksize*suby+1:end,:);
                    LXChunk{param.xchunks,1} = L_tr1(param.chunksize*subx+1:end,:);
                    LYChunk{param.ychunks,1} = L_tr2(param.chunksize*suby+1:end,:);
                    clear X Y L
                     param.M = M;  param.func = func;  param.lambda = lambda(j);
                    param.muta = muta(k);   param.theta=theta(r);   param.nbits = nbits(bi);
                    param.maxItr = maxItr(i); 
                    param.nchunks = min(param.xchunks,param.ychunks);
                    t1 = clock;
                    for chunki = 1:param.nchunks
%                     fprintf('-----chunk----- %3d\n', chunki);
                    if chunki <param.nchunks
                    XTrain_new = XChunk{chunki,:};
                    YTrain_new = YChunk{chunki,:};
                    LXTrain_new = LXChunk{chunki,:};
                    LYTrain_new = LYChunk{chunki,:};
                    else
                    XTrain_new =  cell2mat(XChunk(chunki:end,:));
                    YTrain_new = cell2mat(YChunk(chunki:end,:));
                    LXTrain_new = cell2mat(LXChunk(chunki:end,:));
                    LYTrain_new = cell2mat(LYChunk(chunki:end,:));
                    end
                    GX_new = NormalizeFea(LXTrain_new,1);
                    GY_new = NormalizeFea(LYTrain_new,1);
%                     Hash code learning
                   
                    if chunki == 1
                    
                     [XTrain,YTrain,BBX,BBY,XW,YW,HH] = BMCH0(XTrain_new',YTrain_new',GX_new,GY_new,param);
                     
%                         eva_info_ = evaluate_chunk(XKTrain,YKTrain,LXChunk,LYChunk,XKTest,YKTest,L_te,param,BBX,BBY,XW,YW,chunki);
                    else
                     [BBX,BBY,XW,YW,HH,Q,V] = BMCH(XTrain_new',YTrain_new',GX_new,GY_new,BBX,BBY,HH,param,XTrain,YTrain);
%                         eva_info_ = evaluate_chunk(XKTrain,YKTrain,LXChunk,LYChunk,XKTest,YKTest,L_te,param,BBX,BBY,XW,YW,chunki);

                    end
                    t2 = clock;
                    t(chunki) = etime(t2,t1);
%                       i2t(chunki)=eva_info_.Image_VS_Text_MAP;
%                       t2i(chunki)=eva_info_.Text_VS_Image_MAP;
                     
                   end        
                   BBX = BBX'; BBY = BBY';
                   BX = cell2mat(BBX(:,1:end));
                   BY = cell2mat(BBY(:,1:end));
                  eva_info_ = evaluate_BMCH(XKTrain,YKTrain,L_tr1,L_tr2,XKTest,YKTest,L_te,param,BX,BY);
                    eva_info_.Image_VS_Text_MAP
                    eva_info_.I2Ttop;
                    eva_info_.Text_VS_Image_MAP
                    eva_info_.T2Itop;
                    map(v,1)=eva_info_.Image_VS_Text_MAP;
                    map(v,2)=eva_info_.Text_VS_Image_MAP;
                    top(v,1)=eva_info_.I2Ttop;
                    top(v,2)=eva_info_.T2Itop;
                    result.bits = nbits(bi);
                    result.lambda = maxItr(i);
                    result.muta = muta(k);
                    result.M =theta(r);
                    arry(l,1) = nbits(bi);
                    arry(l,2) = maxItr(i);
                    arry(l,3) = lambda(j);
                    arry(l,4) = muta(k);
                    arry(l,5) = theta(r);
                    arry(l,6) = eva_info_.Image_VS_Text_MAP;
                    arry(l,7) = eva_info_.Text_VS_Image_MAP;
                    l=l+1;   
                roWname={'bits','alpha','beta','lamda','Iter','i2t','t2i'};
                     end
                    fprintf('%d bits average map over %d runs for ImageQueryForText: %.4f\n, top@100 is %.4f\n',nbits(bi), turn, mean(map( : , 1)),mean(top( : , 1)) );
                    fprintf('%d bits average map over %d runs for TextQueryForImage:  %.4f\n, top@100 is %.4f\n',nbits(bi), turn, mean(map( : , 2)),mean(top( : , 2)));
               end
             end
        end
    end
%         xlswrite('covergence.xlsx',arry,'sheel1','A02');
%         save('parameter','GX_new','GY_new','BBX','BBY','Q','V','nbits');
end

  end

