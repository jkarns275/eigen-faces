function y_new=lr_indicator(x_train,y_train,x_test,p)
    % steps: 1) get z 2) get w=y*z'(z*z')-1 3) get y_new=w*z
    z_train = basic_expension(x_train, p);
%     w=z_train\y_train;
%     w = y_train*z_train*inv(z_train'*z_train)
    w = (z_train'*z_train)\(z_train'*y_train);
%     w2= x_train/w'*z_train';
    z_test = basic_expension(x_test, p);
    y_new=w'*z_test';


end