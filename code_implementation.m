
function [mperf, perf, features]=code_implementation(classifier_name, feature_selection_name)

target_variable_index=161;
% Import data
data_file = 'ptsd_dset.txt'; % Insert name of data file
size(data_file)
data_orig =importdata(data_file);
invData = data_orig';
invData = [invData(1:60, :); invData(60: size(invData, 1), :)];
invData = [invData(1:61, :); invData(63: size(invData, 1), :)];
invImputedData = knnimpute(invData);
normAndImputeddata = invImputedData';
[r, c] = size(normAndImputeddata);
data = normAndImputeddata(:,1:c-1);
target = normAndImputeddata(:,c);
for j=1:10
    dataSplits = cvpartition(target ,'KFold', 10); % Splits data for cross-validation
    fprintf('Iteration of cross-validation: %d:',j);
    %% 
    for i=1:10
        fprintf('.');
        % Define training and testing indices
        TEST_indx=dataSplits.test(i)%Define training and testing subsets of data
        TRAIN_indx=dataSplits.training(i);
        trainData=normAndImputeddata(dataSplits.training(i),:); % Change last number to indicate the number of variables (features+target) in the data set
        testData = normAndImputeddata(dataSplits.test(i),:);
        imp_data=knnimpute(trainData');
        imp_data = imp_data';
        size(imp_data)
        imp_test= (knnimpute(testData'))';
        size(imp_test)% Same as above - change to indicate number of variables
        newData=[imp_data; imp_test];
        % Redefine training and testing indices
        TRAIN_indx=1:numel(TRAIN_indx);
        TEST_indx=numel(TRAIN_indx)+1:r;

        % Perform feature selection
        if strcmp(feature_selection_name,'HITON_PC') % Input feature_selection_name 'HITON_PC' if wish to apply parent/children feature selection
            features{i,j}=Causal_Explorer('HITON_PC', trainData, target_variable_index, [], 'z', 0.05, 1);
        elseif strcmp(feature_selection_name,'HITON_MB') % Imput feature_selection_name 'HITON_MB' if wish to apply Markov Blanket feature selection
            features{i,j}=Causal_Explorer('HITON_MB', trainData, target_variable_index, [], 'z', 0.05, 1);
        end
        % Perform classification        
        if strcmp(classifier_name,'SVM') % Input classification_name 'SVM' to classification using SVM with C=1
            prediction=code_SVM(newData(:,features{i,j}), newData(:,target_variable_index), TRAIN_indx, TEST_indx, 'linear', 1);
        elseif strcmp(classifier_name,'NaiveBayes')
            prediction=NBayes(newData(:,features{i,j}), newData(:,target_variable_index), TRAIN_indx, TEST_indx);
        elseif strcmp(classifier_name,'RF')
            prediction=TB(newData(:,features{i,j}), newData(:,target_variable_index), TRAIN_indx, TEST_indx);
        end
        perf(i,j)=auc(prediction, newData(TEST_indx, target_variable_index));                
    end    
    fprintf(' (AUC %g)\n', mean(perf(:,j)));
end   
mperf=mean(mean(perf));

end





