classdef MLP < handle
    properties
        nlayers;    % Integer, number of layers in the network
        lr=0.1;     % Float, learning rate
        layers;     % Cell array of weight matrices. Biases are embedded here.
        ninputs;
        layer_nets = {};
        layer_outputs = {};
        noutputs;
        act = @tanh;
        dact = @(x) ( 1 - (tanh(x).^2) );
    end
    methods
        function obj = MLP(layer_sizes, lr)
            % layer_sizes should be a list of layer sizes (e.g. for a network with 4 inputs, a hidden layer with 2 nodes, and an output
            % layer with 1 node: [4, 2, 1]). So, first number is the number of inputs and last is the number of outputs, anything
            % else is a hidden layer
            % lr is the learning rate
            n_weight_matrices = length(layer_sizes) - 1;
            obj.layers = {};
            obj.ninputs = layer_sizes(1);
            obj.nlayers = n_weight_matrices;
            for layer = 1:n_weight_matrices
                layer_inpt_dim = layer_sizes(layer);
                layer_outpt_dim = layer_sizes(layer + 1);
                % + 1 to layer_inpt_dim for the bias; - 0.5 so weights are in the range [-0.5, 0.5]
                obj.layers{layer} = normrnd(0, 1, [layer_outpt_dim, layer_inpt_dim + 1]);
            end
        end

        function result = think(mlp, input)
            for i = 1:mlp.nlayers
                % Add an extra input set to one for the bias
                len = prod(size(input));
                in = ones(len + 1, 1);
                in(1:len) = input;
                out = mlp.layers{i} * in;
                mlp.layer_nets{i} = out;
                out = mlp.act(out);
                mlp.layer_outputs{i} = out;
                input = out;
            end
            result = out;
        end

        function learn(mlp, input, target)
            output = mlp.think(input);
            err = output - target;
            
            gradients = {};
            for i = 1:mlp.nlayers
                gradients{i} = mlp.layers{i};
            end
            
            error_signals = {};
            p = err .* mlp.dact(mlp.layer_nets{mlp.nlayers});
            error_signals{mlp.nlayers} = p; 
            
            for i = mlp.nlayers:-1:2
                [n_nodes, n_inputs] = size(gradients{i});
                for node = 1:n_nodes
                    gradient = error_signals{i}(node) * mlp.layer_outputs{i - 1};
                    gradients{i}(node, 1:end-1) = gradient';
                    gradients{i}(node, end) = sum(error_signals{i});
                end
                dact = mlp.dact(mlp.layer_nets{i - 1});
                signal = sum(mlp.layers{i}' * error_signals{i});
                error_signals{i - 1} = dact * signal;
            end

            [n_nodes, n_inputs] = size(gradients{1});
            for node = 1:n_nodes
                gradient = error_signals{1}(node) * input;
                gradients{1}(node, 1:end-1) = gradient';
                gradients{1}(node, end) = sum(error_signals{1});
            end
            
            for i = 1:mlp.nlayers
                mlp.layers{i} = mlp.layers{i} - mlp.lr * gradients{i};
            end
        end
    end
end
