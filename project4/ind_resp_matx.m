function train_y=ind_resp_matx(trainlabel)
    n=size(trainlabel,2);
    class=max(trainlabel);
    train_y=zeros(class,n);
    for i =1:n
        train_y(trainlabel(i),i)=1;

    end
end