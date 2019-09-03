% Calls generateTheta function with settings used in the paper
% options should have the following fields:
%   save_file: where to save the parameters to
%   data_file: where to find the data
%   use_train: set to 1 if we want to generate parameters based on the training set, or 0 if based on test set
%   Any additional fields in options will be passed through to generateTheta itself.
function generateThetaFnc(options)
  % File that we will save the set of target parameters to
  save_file = options.save_file;
  % File that contains the dataset
  data_file = options.data_file;

  % load data and decide what (quantile, rep) values to sweep over
  big = strcmp(data_file, 'data/enron_data.mat') || strcmp(data_file, 'data/imdb_data.mat')
  really_big = strcmp(data_file, 'data/imdb_data.mat');
  if big
    if really_big
      % Due to computational limitations we only consider a few different settings for IMDB
      options.quantile_tape = [0.40 0.50 0.60];
      options.rep_tape = [4, 5];
    else
      % Settings that work well for Enron
      options.quantile_tape = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55];
      options.rep_tape = [1, 2, 3, 5, 8, 12, 18, 25, 33];
    end
  else
    % Settings that work well for MNIST, MNIST17, and Dogfish
    options.quantile_tape = [0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25];
    options.rep_tape = [1, 2, 3, 5, 8, 12, 18, 25];
  end
  load(data_file);

  % use_train = 1: generate target parameters using only the training set
  % use_train = 0: generate target parameters with knowledge of the test set
  if ~isfield(options, 'use_train')
    options.use_train = 0;
  end
  options.prune = 0;
  if options.use_train
    [thetas, biases, train_losses, test_errors, quantiles, reps, options] = generateTheta(X_train, y_train, X_train, y_train, X_test, y_test, options);
  else
    [thetas, biases, train_losses, test_errors, quantiles, reps, options] = generateTheta(X_train, y_train, X_test, y_test, X_test, y_test, options);
  end
  save(save_file, 'options', 'thetas', 'biases', 'train_losses', 'test_errors', 'quantiles', 'reps');
  options.prune = 1;
  [thetas, biases, train_losses, test_errors, quantiles, reps] = pruneTheta(thetas, biases, train_losses, test_errors, quantiles, reps);
  save(sprintf('%s_prune.mat', save_file(1:end-4)), 'options', 'thetas', 'biases', 'train_losses', 'test_errors', 'quantiles', 'reps');
end
