function prediction_c = code_SVM(data, target, TRAIN_indx, TEST_indx, kernel, varargin)

% Define a string with SVM parameters
if strcmp(kernel,'linear')
    options=sprintf('-s 0 -t 0 -c %f', varargin{1});
elseif strcmp(kernel,'poly')
   options=sprintf('-s 0 -t 1 -c %f -d %d -r 1 -g 1', varargin{1}, varargin{2});
elseif strcmp(kernel,'rbf')
    options=sprintf('-s 0 -t 2 -c %f -g %f', varargin{1}, varargin{2});
end
target = ones(size(target, 1), 1);
target(1: 50) = 0;
CVSVMModel = fitcsvm(data, target,'Holdout',0.15,...
    'Standardize',true);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
testInds = test(CVSVMModel.Partition);   % Extract the test indices
XTest = data(testInds,:);
YTest = target(testInds,:);
[label,score] = predict(CompactSVMModel,XTest);
prediction_c = score;
table(YTest(1:10),label(1:10),score(1:10,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'});
figure;
h(1:2) = gscatter(data(:,1),data(:,2),target,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data(cl.IsSupportVector,1),data(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(xTest)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
hold off

end