function mse = test(layers, training_iters)
    source_network = MLP(layers, 0.125 / 2);
    target_network = MLP(layers, 0.125 / 2);
    lr = 0.125;
    for i = 1:training_iters
        target_network.lr = lr;
        input = normrnd(0, 1, [layers(1), 1]);
        expected = source_network.think(input);
        target_network.learn(input, expected);
        err = norm(expected - target_network.think(input))^2;
        lr = err / 8;
    end
    source = source_network;
    target = target_network;

    mse = 0;
    for i = 1:10000
        input = normrnd(0, 1, [layers(1), 1]);
        expected = source_network.think(input);
        got = target_network.think(input);
        mse = mse + norm(expected - got)^2;
    end
    mse = mse / 10000;
end
