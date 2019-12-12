function [eigenfaces,eigenvector,t,v]=dimention_reduction(input,pca_method)
    tic;
    [v, eigenvector]=pca_method(input);
    t=toc;
%     t=t2-t1;
%     k_1 = variance(v,th,0);

%     if self==1
%         PC= eigenvector(:, 1:k_1);
%     else
%         PC= eigenvector(:, 1:k);
%     end
%     PC=eigenvector;
    eigenfaces = uint8(normalize(eigenvector, 0, 255));

%     imshow(reshape(eigenfaces(:,1),112,92))


%     y=PC'*input;
%     construct = PC * y;
%     new_face= uint8(normalize(construct, 0, 255));
%     imshow(reshape(new_face_1(:,1),112,92))


end