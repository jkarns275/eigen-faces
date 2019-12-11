function [eigenfaces,k_1,construct,y]=dimention_reduction(input,pca_method,k,th)
    %% PCA with SVD, each column is a PC
    [v, eigenvector]=pca_method(input);
    k_1 = variance(v,th,1);
    %% SVD: eigen faces
    PC= eigenvector(:, 1:k);
    eigenfaces = uint8(normalize(PC, 0, 255));

%     imshow(reshape(eigenfaces(:,1),112,92))

    %% SVD reconstruction
    y=PC'*input;
    construct = PC * y;
%     new_face= uint8(normalize(construct, 0, 255));
%     imshow(reshape(new_face_1(:,1),112,92))


end