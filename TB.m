% Random forests/treebagger
function [post]=TB(data, target, TRAIN_indx, TEST_indx)
 
tree = ClassificationTree.fit(data(TRAIN_indx), target(TRAIN_indx));
 
[post] = predict(tree(TEST_indx), data(TEST_indx));
end
