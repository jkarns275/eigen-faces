
[trainlabel,traindata] = readtrain(2,1);
[testlabel,testdata] = readtest(2,1);
train_y=ind_resp_matx(trainlabel);
% z = basic_expension(traindata, 3);
%%
y_new=lr_indicator(traindata',train_y',testdata',1);
% y_new=lr_indicator2(traindata',trainlabel,testdata',1);
%%
% y_new=round(y_new);
% err=test_y-y_new;
pred=argmax(y_new);
err=testlabel-pred;

%%

acc_rate=length(find(err==0))/size(testlabel,2);