function data = read_all_data(path)
    % path is the location of the att_faces data set
    data = {};
    for i = 1:40
        data{i} = {};
        for j = 1:10
            impath = sprintf("%s/s%d/%d.pgm", path, i, j);
            im = double(imread(impath));
            im = im ./ 255.0;
            im = im - 0.5;
            data{i}{j} = im;
        end
    end
end
