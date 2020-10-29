% NaiveBayes classification - posterior probabilities
function [post]=NBayes(data, target, TRAIN_indx, TEST_indx)
 
NB=NaiveBayes.fit(data(TRAIN_indx), target(TRAIN_indx));
 
post1=posterior(NB,target(TEST_indx));
post=post1(:,1);
 
end 