function [train, test, train_label, test_label]=read_flower(path)
    tr='train';
    te='test';
    train=[];
    test=[];
    train_label=zeros(1,280)+2;
    test_label=zeros(1,120)+2;
    for i = 1:280
        impath = sprintf('%s/%s/%d.jpg', path, tr, i);
        img = imread(impath);
        img=reshape(img,112*92,1);
        train=horzcat(train,img);
    end
    
    for i = 281:400
        impath = sprintf('%s/%s/%d.jpg', path, te, i);
        img = imread(impath);
        img=reshape(img,112*92,1);
        test=horzcat(test,img);
    end
    train=double(train);
    test=double(test);

end
