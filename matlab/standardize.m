function [ input_std ] = standardize(input)
% standardize Makes the input data to have zero mean and unit variance

    input_std = input - repmat(mean(input, 2), 1, size(input, 2));
    input_std = input_std ./ repmat(std(input_std, 1), size(input_std, 1), 1);
end