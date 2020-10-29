
target_variable_index=161;
% Import data
data_file = 'ptsd_dset.txt'; % Insert name of data file
data_orig =importdata(data_file);
invData = data_orig';
invData = [invData(1:60, :); invData(60: size(invData, 1), :)];
invData = [invData(1:61, :); invData(63: size(invData, 1), :)];
invImputedData = knnimpute(invData);
normAndImputeddata = invImputedData';
[r, c] = size(normAndImputeddata);
data = normAndImputeddata(:,1:c-1);
target = normAndImputeddata(:,c);
