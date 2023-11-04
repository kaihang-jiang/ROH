function evaluation=evaluate_chunk(XKTrain,YKTrain,LXTrain,LYTrain,XKTest,YKTest,LTest,param,BBX,BBY,XW,YW, chunk)
    tic;
    addpath(genpath('./utils/'));
    addpath(genpath('markSchmidt/'));
	% Hash functions learning
    LXTrain = cell2mat(LXTrain(1:chunk,:));
    LYTrain = cell2mat(LYTrain(1:chunk,:));
    BBX = BBX'; BBY = BBY';
   BX = cell2mat(BBX(:,1:end));
   BY = cell2mat(BBY(:,1:end));
    if strcmp(param.func,'linear')
   %% linear function
    tBX = compactbit(sign(XKTest*XW)>=0);    
    tBY = compactbit(sign(YKTest*YW)>=0);
    dBX = compactbit(sign(BX')>=0);
    dBY = compactbit(sign(BY')>=0);
    traintime=toc;
    evaluation_info.trainT=traintime;
	tic;
    end
	% Cross-Modal Retrieval
    DHamm1 = hammingDist(tBX, dBX);
       [~, orderH] = sort(DHamm1, 2);
%         evaluation_info.Image_VS_Text_MAP = mAP(orderH',LTrain,LTest);
        evaluation.Image_VS_Text_MAP = perf_metric4Label(LXTrain,LTest, DHamm1');
%         [evaluation_info.I_VS_T_precision, evaluation_info.I_VS_T_recall] = precision_recall(orderH', LTrain, LTest,24.5);
%      evaluation_info.Image_To_Text_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    DHamm2 = hammingDist(tBY, dBY);
       [~, orderH2] = sort(DHamm2, 2);
%        evaluation_info.Text_VS_Image_MAP = mAP(orderH2',LTrain,LTest);
        evaluation.Text_VS_Image_MAP = perf_metric4Label(LYTrain,LTest,DHamm2');        
%        [evaluation_info.T_VS_I_precision,evaluation_info.T_VS_I_recall] = precision_recall(orderH', LTrain, LTest,24.5);
%      evaluation_info.Text_To_Image_Precision = precision_at_k(orderH', LTrain, LTest,param.top_K);
    compressiontime=toc;
    
%    tUX =sign(XKTest*XW);
%    tVY =sign(YKTest*YW);
%      B1 = sign((bsxfun(@minus,B, mean(B,1))))';
% tB1 = sign((bsxfun(@minus,tUX, mean(B',1))));
% B2 = B1;
% tB2 = sign((bsxfun(@minus,tVY , mean(B',1))));
% sim_it = B1 * tB2'; 
% sim_ti = B2 * tB1';
%   [evaluation_info.mapA,evaluation_info.recallA] = myPr(sim_it,LTrain,LTest,24);
%   [evaluation_info.mapB,evaluation_info.recallB] = myPr(sim_ti,LTrain,LTest,24);
        
      
    evaluation_info.compressT=compressiontime;
end