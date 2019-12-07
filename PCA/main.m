% %% read digit image
% [image,label] = read_hw;
% four=find(label==4);
% images=image(:, four);
% 
% %% show image
% imshow(reshape(images(:,1),28,28));

%% face image
images=read_to_matrix('../att_faces');
data1=double(images);
figure()
imshow(reshape(images(:,1),112,92))

% %% use MATLAB pca
% [coeff,score,latent,~,explained,mu] = pca(data1');
% k = variance(explained, 0.9,1);
% % coeff(1:k,:)*images


%% PCA with SVD, each column is a PC
[v1, eigenvector1]=pca_svd(data1);
% figure();
k1 = variance(v1,0.95,0);

%% SVD: eigen faces
PC1= eigenvector1(:, 1:k1);
eigenfaces1 = uint8(normalize(PC1, 0, 255));
figure()
imshow(reshape(eigenfaces1(:,1),112,92))

%% SVD reconstruction
y1=PC1'*data1;
construct1 = PC1 * y1;
figure()
new_face1= uint8(normalize(construct1, 0, 255));
imshow(reshape(new_face1(:,1),112,92))

%% PCA with eigen, each column is a PC
[v2, eigenvector2]=pca_eigen(data1);
figure()
k2 = variance(v2,0.9,1);

%% eigen faces
PC2= eigenvector2(:, 1:k2);
eigenfaces2 = uint8(normalize(PC2, 0, 255));
figure()
imshow(reshape(eigenfaces2(:,1),112,92))



