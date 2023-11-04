function [BBX,BBY,XW,YW,HH,Q,V,XTrain,YTrain] = BMCH(XTrain_new,YTrain_new,GXTrain_new,GYTrain_new,BBX,BBY,HH,param,XTrain,YTrain)  
   max_iter = param.maxItr;
    muta = param.muta; lambda = param.lambda;
    theta = param.theta; nbits = param.nbits;   
    [~,c] = size(GXTrain_new);
    [d1,n] = size(XTrain_new);
    [d2,n1] = size(YTrain_new);
    GXTrain_new = GXTrain_new';
    GYTrain_new = GYTrain_new';
    %initization
    BX_new = sign(randn(nbits, n)); 
    BY_new = sign(randn(nbits, n1));
    Q_new =  rand(c,nbits);
%     XTrain{1,end+1} = XTrain_new;
%     YTrain{1,end+1} = YTrain_new;
%     XKTrain = cell2mat(XTrain(1:end));
%     YKTrain = cell2mat(YTrain(1:end));
    [Ux, ~, ~] = svd(XTrain_new);
    [Uy, ~, ~] = svd(YTrain_new);
    U_x=Ux(1:nbits,:);
    U_y=Uy(1:nbits,:);
    
    for i = 1:max_iter
        %fprintf('iteration %3d\n', i);
       %% update Bt 
        Jx = (HH{1,4} + BX_new*XTrain_new')*pinv(HH{1,6} + XTrain_new*XTrain_new')*Ux;
         for j=1:d1
              if norm(Jx(:,j),2)~=0 && norm(U_x(:,j),2)~=0
              ax(j) = (Jx(:,j)'*U_x(:,j))/(norm(Jx(:,j))'*norm(U_x(:,j)));
              else
              ax(j) = 0;
              end
         end 

        clear j t r
        
        Jy = (HH{1,5} + BY_new*YTrain_new')*pinv(HH{1,7} + YTrain_new*YTrain_new')*Uy;
       for j=1:d2
               if norm(Jy(:,j),2)~=0 && norm(U_y(:,j),2)~=0
               ay(j) = (Jy(:,j)'*U_y(:,j))/(norm(Jy(:,j))*norm(U_y(:,j)));
               else
               ay(j) = 0;
               end
       end 
        
        Z = lambda*GXTrain_new'*Q_new;
        Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
        [~,Lmd,RR] = svd(Temp);
        idx = (diag(Lmd)>1e-8);
        R = RR(:,idx); R_ = orth(RR(:,~idx));
        P = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (R / (sqrt(Lmd(idx,idx))));
        P_ = orth(randn(n,nbits-length(find(idx==1))));
        U = sqrt(n)*[P P_]*[R R_]';
        V_new = U';  
        
        %% update Q
        Q_new = (lambda*(n+HH{1,10})*eye(c)+muta*nbits^2*(GXTrain_new*GXTrain_new'+GYTrain_new*GYTrain_new'+HH{1,1}))\...
            (lambda*(GXTrain_new*V_new'+HH{1,2})+muta*nbits*(GXTrain_new*BX_new'+GYTrain_new*BY_new'+HH{1,3}));   
       
        %% update B
         BX_new = sign(muta*nbits*Q_new'*GXTrain_new+U_x*diag(ax)*Ux'*XTrain_new);
         BX_new(BX_new==0)=-1;
         BY_new = sign(muta*nbits*Q_new'*GYTrain_new+U_y*diag(ay)*Uy'*YTrain_new);
         BY_new(BY_new==0)=-1;
%         loss = norm(BX_new-U_x*diag(ones(1,d1)-ax)*Ux'*XTrain_new,2);
%         loss1 = norm(BY_new-U_y*diag(ones(1,d2)-ay)*Uy'*YTrain_new,2');
%         P1 = lambda*norm(GXTrain_new-Q_new*V_new,'fro')^2;
%         P2 = muta*(norm(BX_new-nbits*Q_new'*GXTrain_new,'fro')^2 +norm(BY_new-nbits*Q_new'*GYTrain_new,'fro')^2);
%         P3 = loss+loss1;
         
%         f(i)=P1+P2+P3;
%          fprintf('The iteration is : %i and f val is : %f \n', i, f(i));   
    end
    
    H1_new = HH{1,1}+GXTrain_new*GXTrain_new'+GYTrain_new*GYTrain_new';
    H2_new = HH{1,2}+GXTrain_new*V_new';
    H3_new = HH{1,3}+GXTrain_new*BX_new'+GYTrain_new*BY_new';
    H4_new = HH{1,4}+BX_new*XTrain_new';
    H5_new = HH{1,5}+BY_new*YTrain_new';
    H6_new = HH{1,6}+XTrain_new*XTrain_new';
    H7_new = HH{1,7}+YTrain_new*YTrain_new';
    H8_new = HH{1,8}+XTrain_new*BX_new';
    H9_new = HH{1,9}+YTrain_new*BY_new';
    H10_new = HH{1,10}+n;
    
    HH{1,1} = H1_new;
    HH{1,2} = H2_new;
    HH{1,3} = H3_new;
    HH{1,4} = H4_new;
    HH{1,5} = H5_new;
    HH{1,6} = H6_new;
    HH{1,7} = H7_new;
    HH{1,8} = H8_new;
    HH{1,9} = H9_new;
    HH{1,10} = H10_new;
    BBX{end+1,1} = BX_new;
    BBY{end+1,1} = BY_new;
    XW = (H6_new + theta*eye(size(XTrain_new,1))) \ H8_new;
    YW = (H7_new + theta*eye(size(YTrain_new,1))) \ H9_new;
    Q = Q_new;
    V = V_new;
end