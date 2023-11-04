function [map, recallA] = myPr(sim_x, L_tr, L_te, num)
j = 0;
mark=0;
x=length(L_tr);
t=int32(x/num);
for R = 1:t:x
    j = j + 1;
    [~, cat] = size(L_tr);
    multiLabel = cat > 1;
    if multiLabel 
        Label = L_tr * L_te';
    end
    tn = size(sim_x,2);  % query 的样本数
    ap = zeros(tn,1);
    recall = zeros(tn,1);
    for i = 1 : tn
        if mark == 0
            % inxx 保存与第 i 个测试样本 hammingDist 最小的前 R 个 database 样本所在的位置
            [~, inxx] = sort(sim_x(:, i), 'descend');
        elseif mark == 1
            [~, inxx] = sort(sim_x(:, i));
        end

        if multiLabel
           inxx = inxx(1: R);  
           ranks = find(Label(inxx, i) > 0)';
        else
           inxx = inxx(1: R);
           tr_gt = L_tr(inxx);  % tr_gt 为前 R 个实例的标签
           ranks = find(tr_gt == L_te(i))';  % ranks 为 groundtrue
        end 
        % compute AP for the query
        if isempty(ranks)
            ap(i) = 0;
        else
            % ap(i) = sum((1: length(ranks)) ./ ranks) / length(ranks);
            if multiLabel
                % #relavant-in-result / #result
                ap(i) = length(ranks) / length(inxx);
                % #relavant-in-result / #relavant-in-all
                recall(i) = length(ranks) / length(find(Label(:, i)>0));
            else
                ap(i) = length(ranks) / length(inxx);
                recall(i) = length(ranks) / length(find(L_tr == L_te(i)));
            end
            
        end
    end
    map(j) = roundn(mean(ap),-5);
    recallA(j) = roundn(mean(recall),-5);
end