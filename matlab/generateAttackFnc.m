function generateAttackFnc(dataset, epsilon, att_rep, grad, options)
  % default options struct if none is specified
  if nargin < 5
    options = struct();
    options.LP = 1;
    options.timeit = 0;
    options.nips = 0;
    options.use_train = 0;
    options.slab = 0;
  end
  fprintf(1, 'generating attack: %s %f %d %d\n', dataset, epsilon, att_rep, grad);
  % Generate attack designed to get past data sanitization that throws away 10% of most outlying points
  options.p = 0.10;
  % Actually evaluate against data sanitization that throws away 5% of most outlying points
  options.p_def = 0.05;
  % Evaluate each defense once (vs. multiple times to account for randomness)
  options.evaluation_trials = 1;
  % Use hinge loss, solve the SVM exactly
  options.loss = 'hinge';
  options.method = 'exact';
  options.num_rounds = 1;
  % Repeat each attack point att_rep times (gets past nearest neighbor defense and makes some attacks more effective)
  options.att_rep = att_rep;
  % If we're computing timing statistics, then we should skip target parameters that seem unpromising and 
  % also avoid running expensive diagnostics
  if options.timeit
    options.skip = 1;
    options.def_ids = 1;
    options.ordering = 'sqrt';
  else
    options.skip = 0;
  end
  % Sort attacks based on how well they do in the absence of any defense
  % (Note that the attacks are optimized to get past most of the defenses, so this 
  %  number is usually similar to the number against a defense.)
  options.aggregator = @(s)median(s.none);
  options.save_all = 1;
  if grad
    options.guarder = @(t){0*t,t};
  else
    options.guarder = @(t){0*t};
  end
  % Set weight decay, number of burn-in samples, etc. for specific datasets
  % target_file specifies where we load the target parameters from
  switch dataset
    case 'enron'
      data_file = 'data/enron_data.mat';
      if options.use_train
        target_file = 'data/enron_thetas_with_bias_exact_decay_09_use_train_v3.mat';
      else
        target_file = 'data/enron_thetas_with_bias_exact_decay_09_v3_prune.mat';
      end
      options.decay = 0.09;
      options.burn_frac = max(1.0, 0.10/epsilon-1);
    case 'imdb'
      data_file = 'data/imdb_data.mat';
      assert(~options.use_train);
      target_file = 'data/imdb_thetas_with_bias_exact_decay_01_all_v3_prune.mat';
      options.decay = 0.01;
      options.burn_frac = max(1.00, 0.02/epsilon-1);
      % MATLAB evaluation is too expensive to run at all for IMDB (we run it in python instead)
      options.evaluation_trials = 0;
    case 'dogfish'
      data_file = 'data/dogfish_data.mat';
      assert(~options.use_train);
      target_file = 'data/dogfish_thetas_with_bias_exact_decay_110_v3_prune.mat';
      options.decay = 1.10;
      options.burn_frac = max(1.0, 0.10/epsilon-1);
    case 'mnist17'
      data_file = 'data/mnist_17.mat';
      assert(~options.use_train);
      target_file = 'data/mnist_17_thetas_with_bias_exact_decay_01_v3_prune.mat';
      options.decay = 0.01;
      options.burn_frac = max(0.33, 0.02/epsilon-1);
    case 'synthetic'
      data_file = 'data/synthetic_data.mat';
      assert(~options.use_train);
      target_file = 'data/synthetic_thetas_with_bias_exact_decay_005_v3_prune.mat';
      options.decay = 0.005;
      options.burn_frac = max(0.5, 0.05/epsilon-1);
    case 'mnist'
      data_file = 'data/mnist_data.mat';
      assert(~options.use_train);
      target_file = 'data/mnist_thetas_with_bias_adagrad_decay_00_v3_prune.mat';
      % Solving the LP exactly is too expensive on MNIST so we use adagrad instead
      % (and set weight decay to 0 since adagrad already implicitly regularizes)
      options.method = 'adagrad';
      options.decay = 0.00;
      options.burn_frac = max(0.33, 0.015/epsilon-1);
    otherwise
      assert(false);
  end
  % the min-max attack in the old NIPS submission didn't use any target parameters
  if options.nips
    target_file = 'data/no_target.mat';
    options.guarder = @(t){zeros(5116,1)};
  end
  % set name of where we save data to based on the options
  if grad
    diary_file = sprintf('diaries/%s/eps-%d_wd-%d_exact_L2_grad_rep-%d_all.log', dataset, round(1000*epsilon), round(100*options.decay), att_rep);
  else
    diary_file = sprintf('diaries/%s/eps-%d_wd-%d_exact_L2_rep-%d_all.log', dataset, round(1000*epsilon), round(100*options.decay), att_rep);
  end
  if options.timeit
    diary_file = sprintf('%s_timeit.log', diary_file(1:end-4));
  end
  if options.LP == 0
    diary_file = sprintf('%s_noLP.log', diary_file(1:end-4));
  end
  if options.nips
    diary_file = sprintf('%s_nips_p-%d.log', diary_file(1:end-4), round(100*options.p));
  end
  if options.use_train
    diary_file = sprintf('%s_use_train.log', diary_file(1:end-4));
  end
  if options.slab
    diary_file = sprintf('%s_slab_fixed.log', diary_file(1:end-4));
  end
  generateAttackAll(diary_file, data_file, target_file, epsilon, options);
end
