function [eigenfaces,k_1,construct,y,t]=dimention_reduction(input,pca_method,k,th,self)
    tic;
    %% PCA with SVD, each column is a PC
    [v, eigenvector]=pca_method(input);
    t=toc;
%     t=t2-t1;
    k_1 = variance(v,th,1);
    %% SVD: eigen faces
    if self==1
        PC= eigenvector(:, 1:k_1);
    else
        PC= eigenvector(:, 1:k);
    end
    eigenfaces = uint8(normalize(PC, 0, 255));

%     imshow(reshape(eigenfaces(:,1),112,92))

    %% SVD reconstruction
    y=PC'*input;
    construct = PC * y;
%     new_face= uint8(normalize(construct, 0, 255));
%     imshow(reshape(new_face_1(:,1),112,92))


end