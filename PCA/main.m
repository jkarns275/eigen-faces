%% read digit image
[image,label] = read_hw;
four=find(label==4);
images=image(:, four);

%% show image
imshow(reshape(images(:,1),28,28));

%% face image
data1=read_to_matrix('../att_faces');
data1=double(data1);
%% use MATLAB pca
[coeff,score,latent,~,explained,mu] = pca(data1');
k = variance(explained, 0.7);
% coeff(1:k,:)*images


%% PCA with SVD, each column is a PC
[v1, eigenvector1]=pca_svd(data1);
k1 = variance(v1,0.7);

%% eigen faces
% eigenfaces1 = eigenvector1 * data1;
eigenfaces1 = data1 * eigenvector1;
% eigenfaces1 = eigenvector1' * data1';
eigenfaces1 = uint8(normalize(eigenfaces1, 0, 255));
imshow(reshape(eigenfaces1(:,2),112,92))

%% PCA with eigen, each column is a PC
[v2, eigenvector2]=pca_eigen(data1);
k2 = variance(v2,0.7);

%% eigen faces
eigenfaces2 = eigenvector2' * data1;
% eigenfaces2=int(eigenfaces2);
imshow(reshape(eigenfaces2(:,2),112,92))
eigenfaces1 = uint8(normalize(eigenfaces1, 0, 255));
imshow(reshape(eigenfaces1(:,2),112,92))