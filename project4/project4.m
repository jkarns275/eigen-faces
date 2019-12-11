

%% face image
[train_data,train_label,test_data,test_label,leftout_data,leftout_label]=get_train_test('../att_faces');
input_data=horzcat(train_data, test_data);

%% dimention reduction
k=100;
th=0.9;
[eigenfaces_1,k_1,construct_1,y_1]=dimention_reduction(input_data,@pca_svd,k,th);
[eigenfaces_2,k_2,construct_2,y_2]=dimention_reduction(input_data,@pca_eigen,k,th);
%% compare constructed and original
d=zeros(1,350);
for i = 1: 350
    d(i)= norm(input_data(:,i)-construct_1(:,i));
end
%% face recognition
[flower_train, flower_test, flower_train_label, flower_test_label]=read_flower('../tulip');
flower_face_train=horzcat(train_data, flower_train);
flower_face_test=horzcat(horzcat(test_data, leftout_data),flower_test);
flower_face=horzcat(flower_face_train, flower_face_test);
ff_train_label=horzcat(ones(1,280),flower_train_label);
ff_test_label=horzcat(ones(1,120),flower_test_label);
k=120;
th=0.9;
[eigenfaces_f,k_f,construct_f,y_f]=dimention_reduction(flower_face,@pca_svd,k,th);
acc_rate_f=classification(y_f,size(flower_face_train,2), ff_train_label, ff_test_label);

%% classification
acc_rate_1=classification(y_1,size(train_data,2), train_label, test_label, 35);
acc_rate_2=classification(y_1,size(train_data,2), train_label, test_label, 35);
clc
