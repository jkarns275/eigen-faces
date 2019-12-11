function [train_data,train_label,test_data,test_label,leftout_data,leftout_label]=get_train_test(path)
    test_data=[];
    test_label=[];
    train_data=[];
    train_label=[];
    leftout_data=[];
    leftout_label=[];
    for i = 1:35
        for j = 1:8
            impath = sprintf('%s/s%d/%d.pgm', path, i, j);
            im = imread(impath);
            im_row=reshape(im,112*92,1);
            train_data=horzcat(train_data, im_row);
            train_label=horzcat(train_label, i);

        end
        for j = 9:10
            impath = sprintf('%s/s%d/%d.pgm', path, i, j);
            im = imread(impath);
            im_row=reshape(im,112*92,1);
            test_data=horzcat(test_data, im_row);
            test_label=horzcat(test_label, i);

        end
    end
     for i = 36:40
        for j = 1:10
            impath = sprintf('%s/s%d/%d.pgm', path, i, j);
            im = imread(impath);
            im_row=reshape(im,112*92,1);
            leftout_data=horzcat(leftout_data, im_row);
            leftout_label=horzcat(leftout_label, i);

        end
     end
    test_data=double(test_data);
    test_label=double(test_label);
    train_data=double(train_data);
    train_label=double(train_label);
    leftout_data=double(leftout_data);
    leftout_label=double(leftout_label);
end