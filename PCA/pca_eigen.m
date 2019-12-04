function [eigenvalue, eigenvector]=pca_eigen(images)
% this function doesn't return correct values
    avg=mean(images,2);
    [m,n]=size(images);
    sub_avg=images-repmat(avg,1,n);
%     v=cov(sub_avg');
    covariance = (1 / (n-1)) *( sub_avg * sub_avg');
%     p=v-covariance;% v=covariance
    [values, vectors] = eig(covariance);
    d1=diag(values);
    [eigenvalue, index]=sort(d1,'descend');
    eigenvector=vectors(:,index);
%     p = variance(eigenvalue,th);

end 
   
  


