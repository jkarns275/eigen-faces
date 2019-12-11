function acc_rate=classification(y,t_size, train_label, test_label)
    y_train=y(:, 1:t_size);
    y_test=y(:, (t_size+1):size(y,2));
    train_y=ind_resp_matx(train_label);
    % acc=zeros(1,20);
% for p=1:20
    y_new=lr_indicator(y_train',train_y',y_test',1);
    
    pred=argmax(y_new);
    err=test_label-pred;
    
    acc_rate=length(find(err==0))/size(test_label,2);
%     acc(1,p)=acc_rate;
% end
% plot(acc)


end