function evaluation_info=evaluate_BMCH(XKTrain,YKTrain,LXTrain,LYTrain,XKTest,YKTest,LTest,param,BX,BY)
    tic;
    addpath(genpath('./utils/'));
    addpath(genpath('markSchmidt/'));
	% Hash functions learning
    
   %% linear regression
    XW =pinv(XKTrain'*XKTrain+param.theta*eye(size(XKTrain,2)))   *   (XKTrain'*BX');
    YW =pinv(YKTrain'*YKTrain+param.theta*eye(size(YKTrain,2)))   *   (YKTrain'*BY');
   
    tBX = compactbit(sign(XKTest*XW)>=0);    
    tBY = compactbit(sign(YKTest*YW)>=0);
    dBX = compactbit(sign(BX')>=0);
    dBY = compactbit(sign(BY')>=0);
    traintime=toc;
    evaluation_info.trainT=traintime;
	tic;

%% Cross-Modal Retrieval
    DHamm1 = hammingDist(tBX, dBX);
       [~, orderH] = sort(DHamm1, 2);
         evaluation_info.Image_VS_Text_MAP = perf_metric4Label(LXTrain,LTest, DHamm1');
%            [evaluation_info.I_VS_T_precision, evaluation_info.I_VS_T_recall] = precision_recall(orderH', LXTrain, LTest);
%            evaluation_info.I_VS_T_TopK=precision_at_k(orderH', LXTrain, LTest,2001,20);
             evaluation_info.I2Ttop=mean(precision_at_k(orderH', LXTrain, LTest,100,100));
    DHamm2 = hammingDist(tBY, dBY);
       [~, orderH2] = sort(DHamm2, 2);
         evaluation_info.Text_VS_Image_MAP = perf_metric4Label(LYTrain,LTest,DHamm2');        
%           [evaluation_info.T_VS_I_precision,evaluation_info.T_VS_I_recall] = precision_recall(orderH2', LYTrain, LTest);
%             evaluation_info.T_VS_I_TopK=precision_at_k(orderH2', LYTrain, LTest,2001,20);
         evaluation_info.T2Itop=mean(precision_at_k(orderH2', LYTrain, LTest,100,100));
    compressiontime=toc;
        
      
    evaluation_info.compressT=compressiontime;
    clear B XW YW
end