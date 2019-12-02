function mse = test(att_faces_data, training_iters)
    im = att_faces_data{1}{1};
    [rows, cols] = size(im);
    data_dim = im(:);

    layers = [rows * cols, rows * cols / 2, rows * cols / 4, rows * cols / 2, rows * cols];

    target_network = MLP(layers, 0.125 / 2);
    lr = 10;

    for i = 1:training_iters
        for face_index = 1:10
            for subject = 1:40
                input = att_faces_data{subject}{face_index}(:);
                target_network.learn(input, input);
                err = norm(input - target_network.think(input));
                fprintf("Error = %f\n", err);
                lr = err / 8;
            end
        end
    end
end
