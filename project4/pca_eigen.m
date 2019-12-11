function [eigenvalue, eigenvector]=pca_eigen(images)

    avg=mean(images,2);
    [m,n]=size(images);
    sub_avg=images-repmat(avg,1,n);
%     v=cov(sub_avg');
%     covariance = (1 / sqrt(n-1)) *( sub_avg * sub_avg');
    covariance = ( sub_avg * sub_avg');
%     p=v-covariance;% v=covariance
    [vectors, values ] = eig(covariance);
    d1=diag(values);
    [eigenvalue, index]=sort(d1,'descend');
    eigenvector=vectors(:,index);
%     p = variance(eigenvalue,th);

end 
   
  


