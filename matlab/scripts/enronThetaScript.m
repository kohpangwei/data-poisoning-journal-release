function enronThetaScript()
  options = struct();
  options.data_file = 'data/enron_data.mat';
  options.loss = 'hinge';
  options.method = 'exact';
  options.decay = 0.09;

  options.use_train = 0;
  options.save_file = 'data/enron_thetas_with_bias_exact_decay_09_v3.mat';
  generateThetaFnc(options)

  options.use_train = 1;
  options.save_file = 'data/enron_thetas_with_bias_exact_decay_09_use_train_v3.mat';
  generateThetaFnc(options)
end
