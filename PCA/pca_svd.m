function [variances, eigenvector]=pca_svd(images)

    avg=mean(images,2);
    [m,n]=size(images);

    sub_avg = images-repmat(avg,1,n);
%     Y = sub_avg' / sqrt(n-1);
%     [~,s,eigenvector] = svd(Y);
    [~,s,eigenvector] = svd(sub_avg);
    s_diag=diag(s);
    variances = s_diag.^2;
%     p = variance(sig2,th);
  
end 
   
  


