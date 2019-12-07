function data=read_to_matrix(path)
    data=[];
    for i = 1:40
        for j = 1:10
            impath = sprintf('%s/s%d/%d.pgm', path, i, j);
            im = imread(impath);
            im_row=reshape(im,112*92,1);
            data=horzcat(data, im_row);

        end
    end
end