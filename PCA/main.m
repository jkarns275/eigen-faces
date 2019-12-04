%% read image
[image,label] = read_hw;
four=find(label==4);
images=image(:, four);

%% show image
imshow(reshape(images(:,1),28,28));

%% use MATLAB pca
[coeff,score,latent,tsquared,explained,mu] = pca(images');
k = variance(explained, 0.7);
% coeff(1:k,:)*images


%% PCA with SVD, each column is a PC
[v1, eigenvector1]=pca_svd(images);
k1 = variance(v1,0.7);

%% PCA with eigen, each column is a PC
[v2, eigenvector2]=pca_eigen(images);
k2 = variance(v2,0.7);