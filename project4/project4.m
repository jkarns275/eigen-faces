function [acc_f, acc_c, avg_d, k_variance]=project4    
%% Dimension Reduction
    images=read_to_matrix('../att_faces');
    data_400=double(images);

    th=[0.8, 0.85, 0.9, 0.95];
    k_variance=zeros(2,4);

    avg_d=zeros(2,4);
    face1=[];
    face2=[];
    
    [eigenfaces_1,eigenvector_1,t_1,v_1]=dimention_reduction(data_400,@pca_svd);
    [eigenfaces_2,eigenvector_2,t_2,v_2]=dimention_reduction(data_400,@pca_eigen);
    for i=1:4
        k_variance(1,i) = variance(v_1,th(i),0);
        k_variance(2,i) = variance(v_2,th(i),0);
        [construct_1,~]=reconstruction(data_400,k_variance(1,i),eigenvector_1);
        [construct_2,~]=reconstruction(data_400,k_variance(2,i),eigenvector_2);

        new_face_1= uint8(normalize(construct_1, 0, 255));
        new_face_2= uint8(normalize(construct_2, 0, 255));
        face1=horzcat(face1,new_face_1(:,1));
        face2=horzcat(face2,new_face_2(:,1));
        d1=zeros(1,400);
        d2=zeros(1,400);
        for j = 1: 400
            d1(j)=norm(construct_1(:,j)-data_400(:,j));
            d2(j)=norm(construct_1(:,j)-data_400(:,j));
        end
        avg_d(1,i)=mean(d1);
        avg_d(2,i)=mean(d2);
    end
% reconstruct face
    figure()
    for i=1:4
        subplot(2,5,i);
        imshow(reshape(face1(:,i),112,92));
        subplot(2,5,i+5);
        imshow(reshape(face2(:,i),112,92));
    end
    subplot(2,5,5);
    imshow(reshape(images(:,1),112,92))
    subplot(2,5,10);
    imshow(reshape(images(:,1),112,92))
% eigenfaces
    figure()
    for i = 1:8
        subplot(2,8,i);
        imshow(reshape(eigenfaces_1(:,i),112,92));
        subplot(2,8,i+8);
        imshow(reshape((eigenfaces_2(:,i)),112,92));
    end


    %% classification
    [train_data,train_label,test_data,test_label,leftout_data,leftout_label]=get_train_test('../att_faces');

        %% face recognition
    [flower_train, flower_test, flower_train_label, flower_test_label]=read_flower('../tulip');
    flower_face_train=horzcat(train_data, flower_train);
    flower_face_test=horzcat(horzcat(test_data, leftout_data),flower_test);
    flower_face=horzcat(flower_face_train, flower_face_test);
    ff_train_label=horzcat(ones(1,280),flower_train_label);
    ff_test_label=horzcat(ones(1,120),flower_test_label);
    [~,eigenvector_f,~,~]=dimention_reduction(flower_face,@pca_svd);
    
%     acc_rate_f=zeros(1,160);
%     for k= 1:160
%         [~,y_f]=reconstruction(flower_face,k,eigenvector_f);
%         acc_rate_f(k)=classification(y_f,size(flower_face_train,2), ff_train_label, ff_test_label);
%     end
%     [acc_f,k_f]=max(acc_rate_f);
%     plot(acc_rate_f);
    k_f=19;
    [~,y_f]=reconstruction(flower_face,k_f,eigenvector_f);
    acc_f=classification(y_f,size(flower_face_train,2), ff_train_label, ff_test_label);
        %% face identification
%     acc_rate_c=zeros(1,120);
%     input_data=horzcat(train_data, test_data);
%     [~,eigenvector_c,~,~]=dimention_reduction(input_data,@pca_svd);
%     for k=1:110
%         [~,y_c]=reconstruction(input_data,k,eigenvector_c);
%         acc_rate_c(k)=classification(y_c,size(train_data,2), train_label, test_label);
%     end
%     [acc_c,k_c]=max(acc_rate_c);
%     plot(acc_rate_c);
    
    k_c=47;
    input_data=horzcat(train_data, test_data);
    [~,eigenvector_c,~,~]=dimention_reduction(input_data,@pca_svd);
    [~,y_c]=reconstruction(input_data,k_c,eigenvector_c);
    acc_c=classification(y_c,size(train_data,2), train_label, test_label);
end