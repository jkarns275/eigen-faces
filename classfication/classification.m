% %%
% [trainlabel,traindata] = readtrain(2,1);
% [testlabel,testdata] = readtest(2,1);
% train_y=ind_resp_matx(trainlabel);
% z = basic_expension(traindata, 3);

%% face
% read images and devide into training and testing data (double)
[train_data,train_label,test_data,test_label,leftout_data,leftout_label]=get_train_test('../att_faces');

train_y=ind_resp_matx(train_label);
%%
acc=zeros(1,10);
for p=1:10
    y_new=lr_indicator(train_data',train_y',test_data',p);
    % y_new=lr_indicator2(traindata',trainlabel,testdata',1);
    
    % y_new=round(y_new);
    % err=test_y-y_new;
    pred=argmax(y_new);
    err=test_label-pred;
    
    acc_rate=length(find(err==0))/size(test_label,2);
    acc(1,p)=acc_rate;
end
plot(acc)