function project4    
%% Dimension Reduction
    images=read_to_matrix('../att_faces');
    data_400=double(images);

    th=[0.8, 0.85, 0.9, 0.95];
    k_variance=zeros(2,4);
    t=zeros(2,4);
    avg_d=zeros(2,4);

    for i=1:4
        [eigenfaces_1,k_1,construct_1,~,t_1]=dimention_reduction(data_400,@pca_svd,0,th(i),1);
        [eigenfaces_2,k_2,construct_2,~,t_2]=dimention_reduction(data_400,@pca_eigen,0,th(i),1);
        k_variance(1,i)=k_1;
        k_variance(2,i)=k_2;
        t(1,i)=t_1;
        t(2,i)=t_2;
        new_face_1= uint8(normalize(construct_1, 0, 255));
        new_face_2= uint8(normalize(construct_2, 0, 255));
        face1=horzcat(face1,new_face_1(:,1));
        face2=horzcat(face2,new_face_2(:,1));
        d1=zeros(1,400);
        d2=zeros(1,400);
        for j = 1: 400
            d1(j)=norm(construct_1(:,j)-data_400(:,j));
            d2(j)=norm(construct_2(:,j)-data_400(:,j));
        end
        avg_d(1,i)=mean(d1);
        avg_d(2,i)=mean(d2);
    end

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

    figure()
    for i = 1:4
    subplot(2,4,i);
    imshow(reshape(eigenfaces_1(:,i),112,92));
    end
    for i = 5:8
    subplot(2,4,i);
    imshow(reshape(eigenfaces_2(:,i-4),112,92));
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

    k=130;
    [~,~,~,y_f]=dimention_reduction(flower_face,@pca_svd,k,0,0);
    acc_rate_f=classification(y_f,size(flower_face_train,2), ff_train_label, ff_test_label);

        %% face identification
    k=90;
    input_data=horzcat(train_data, test_data);
    [~,~,~,y_c]=dimention_reduction(input_data,@pca_svd,k,0,0);
    acc_rate_c=classification(y_c,size(train_data,2), train_label, test_label);

end