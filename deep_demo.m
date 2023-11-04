close all; clear; clc;
addpath(genpath('./utils/'));
addpath(genpath('./codes/'));

result_URL = './results/';
if ~isdir(result_URL)
    mkdir(result_URL);
end
tic;
turn = 1;
for v = 1:turn
   db = {'NUSWIDE21'};
    param.top_K = 2000;
    for dbi = 1     :length(db)
    db_name = db{dbi}; param.db_name = db_name;
    nbits = [32];     
    func = 'linear'; 
    %% load dataset
    load(['./datasets/',db_name,'_deep.mat'])
    result_name = [result_URL 'deep_' db_name '_result' '.mat'];
    if strcmp(db_name, 'MIRFLICKR')
     maxItr = [20]; 
    lambda = [10000];
      muta = [1];
     theta = [0.01];
        R = randperm(size(X,1));
        queryInds = R(1:2100); 
        sampleInds = R(2101:2101+10500);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);

    elseif strcmp(db_name, 'NUSWIDE21')
         maxItr = [20]; 
    lambda = [10000];
      muta = [1];
     theta = [0.01];
        R = randperm(size(X,1));
        queryInds = R(1:2100);
        sampleInds = R(2101:2101+10500);
        XTrain = X(sampleInds, :); YTrain = Y(sampleInds, :); LTrain = L(sampleInds, :);
        XTest = X(queryInds, :); YTest = Y(queryInds, :); LTest = L(queryInds, :);
    end
    
    clear X Y L PCA_Y R XAll
    

%% parameter setting 

%         d = [0 2 4 6 8 10 12 14 16 18 20];
        l = 1; %excel writing parameter0
%  muta = [0.001,0.01,0.1,1,10,100, 1000, 10000,100000, 1000000, 10000000];
%          lambda = [0.001,0.01,0.1,1,10,100, 1000, 10000,100000, 1000000,10000000];
%           theta = [0.00000001, 0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1,10,100];
 %% start
for bi = 1:length(nbits)
    for i = 1:length(maxItr)
        for j=1:length(lambda)
             for k = 1:length(muta)
               for r=1:length(theta)

                          %% Kernel representation
                   BMCHparam.nAnchors = 1500; 
                   [XKTrain,XKTest] = Kernelize(XTrain,XTest,BMCHparam.nAnchors); 
                   [YKTrain,YKTest] = Kernelize(YTrain,YTest,BMCHparam.nAnchors);
                   XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1));     
                   XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));    
                   YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1));     
                   YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));  

                %% Label Format
                if isvector(LTrain)
                    LTrain = sparse(1:length(LTrain), double(LTrain), 1); LTrain = full(LTrain);
                    LTest = sparse(1:length(LTest), double(LTest), 1); LTest = full(LTest);
                end
            %% load data
              param.chunksize = 1000;
              R = randperm(size(LTrain,1));
              sampleInds = R(1:end);
              param.nchunks = floor(length(sampleInds)/param.chunksize);

                    XChunk = cell(param.nchunks,1);
                    YChunk = cell(param.nchunks,1);
                    LChunk = cell(param.nchunks,1);
                    for subi = 1:param.nchunks-1
                        XChunk{subi,1} = XKTrain(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                        YChunk{subi,1} = YKTrain(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                        LChunk{subi,1} = LTrain(sampleInds(param.chunksize*(subi-1)+1:param.chunksize*subi),:);
                    end
                    XChunk{param.nchunks,1} = XKTrain(sampleInds(param.chunksize*subi+1:end),:);
                    YChunk{param.nchunks,1} = YKTrain(sampleInds(param.chunksize*subi+1:end),:);
                    LChunk{param.nchunks,1} = LTrain(sampleInds(param.chunksize*subi+1:end),:);

                    XTest = XKTest; YTest = YKTest; LTest = LTest;
                    clear X Y L
                    param.maxItr = maxItr(i);   param.func = func;  param.lambda = lambda(j);
                    param.muta = muta(k);   param.theta=theta(r);   param.nbits = nbits(bi);
                    for chunki = 1:param.nchunks
                    fprintf('-----chunk----- %3d\n', chunki);
        
                    LTrain = cell2mat(LChunk(1:chunki,:));
                    XTrain_new = XChunk{chunki,:};
                    YTrain_new = YChunk{chunki,:};
                    LTrain_new = LChunk{chunki,:};
                    G_new = NormalizeFea(LTrain_new,1);
                    % Hash code learning
                    if chunki == 1
                     tic;
                   [XTrain,YTrain,BBX,BBY,XW,YW,HH] = BMCH0(XTrain_new',YTrain_new',G_new,G_new,param);
%                      eva_info_ = evaluate_chunk(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,param,BB,XW,YW);
%                      i2t(chunki)=eva_info_.Image_VS_Text_MAP;
%                      t2i(chunki)=eva_info_.Text_VS_Image_MAP;
                     traintime=toc;  % Training Time
                     evaluation_info.trainT=traintime;
                    else
                     tic;
                     [BBX,BBY,XW,YW,HH,Q,V] = BMCH(XTrain_new',YTrain_new',G_new,G_new,BBX,BBY,HH,param,XTrain,YTrain);
                     traintime=toc;  % Training Time
                     evaluation_info.trainT=traintime;
%                      eva_info_ = evaluate_chunk(XKTrain,YKTrain,LTrain,XKTest,YKTest,LTest,param,BB,XW,YW);
%                      i2t(chunki)=eva_info_.Image_VS_Text_MAP;
%                      t2i(chunki)=eva_info_.Text_VS_Image_MAP;
                    end
                    end
                     BBX = BBX'; BBY = BBY';
                   BX = cell2mat(BBX(:,1:end));
                   BY = cell2mat(BBY(:,1:end));
                   eva_info_ = evaluate_BMCH(XKTrain,YKTrain,LTrain,LTrain,XKTest,YKTest,LTest,param,BX,BY,XW,YW);
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
                % roWname={'bits','alpha','beta','lamda','Iter','i2t','t2i'};
                  end
                  toc
               end
             end
        end
    end
%        xlswrite('mirflickr.xlsx',arry,'sheel1','A02');
%       save('parameter','XKTest','YKTest','W_1','W_2','B');
end
end
 fprintf('%d bits average map over %d runs for ImageQueryForText: %.4f\n, var is %.4f\n',nbits(bi), turn, mean(map( : , 1)), std(map( : , 1).*1000));
 fprintf('%d bits average map over %d runs for TextQueryForImage:  %.4f\n, var is %.4f\n',nbits(bi), turn, mean(map( : , 2)), std(map( : , 2).*1000));