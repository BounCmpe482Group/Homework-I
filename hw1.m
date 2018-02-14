clc ; clear ; close all;
load("hw1_data.mat");
X_orig = X;
%% Pre-process Data
nonzero_idx = (X ~= 0);
X(nonzero_idx) = log(X(nonzero_idx)) + 1  ;

%%%%%%% sum directly
Z = sum(X, 3);


Y(Y>0) = 1;
Y = Y(:);

%% Best-rank K Approximation
[U, S, V] = svd(Z);
figure('Name' , 'Following the instructions', 'NumberTitle' , 'off');
k_values = [2 10 20 50 100 300];
for i = 1:length(k_values)
    k = k_values(i);
    apprZ = U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
	apprZ = apprZ(:);
    [A, B, T, auc] = perfcurve(Y, apprZ, 1);
    subplot(2,3,i);
    plot(A,B);
    title({['k = ',num2str(k)] ; ['AUC =', num2str(auc)]});
    xlabel('False positive rate'); ylabel('True positive rate');
end

%% Plot singular values
figure('Name' , 'Singular values of Z', 'NumberTitle' , 'off');
plot(diag(S) , '*');


%% BONUS
X = X_orig;
nonzero_idx = (X ~= 0);
X(X>1) = 1;

%%%%%%% cumulative sum of coefficients : 1,3,6,10,...
[s1, s2, s3] = size(X);
Z = reshape(reshape(X, numel(X), 1, 1).*repelem(cumsum(1:s3), s1*s2)' , size(X));
Z = sum(Z,3);
%%%%%%%
%% Best-rank K Approximation
[U, S, V] = svd(Z);
figure('Name' , 'BONUS', 'NumberTitle' , 'off');
for i = 1:length(k_values)
    k = k_values(i);
    apprZ = U(:,1:k)*S(1:k,1:k)*V(:,1:k)';
	apprZ = apprZ(:);
    [A, B, T, auc] = perfcurve(Y, apprZ, 1);
    subplot(2,3,i);
    plot(A,B);
    title({['k = ',num2str(k)] ; ['AUC =', num2str(auc)]});
    xlabel('False positive rate'); ylabel('True positive rate');
end