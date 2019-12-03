function pca(im)
%     data=reshape(im,[28,28,5842]);
%     for i = 1
%         imshow(squeeze(data(:,:,i)))
%     end
    im=im(:,1: 1000);
    avg=mean(im);
    sub_avg=im-avg;
    [u,s,v] = svd(sub_avg);
    s_diag=diag(s);
    sig2 = s_diag.^2;
    plot(sig2)
    


end