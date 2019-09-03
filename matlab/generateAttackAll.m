function [X_attack_best, y_attack_best, best_error]=generateAttackAll(diary_file, data_file, target_file, epsilon, options)

  start = systime;

  options = processOptionsGenerateAttackAll(options);
  diary(diary_file);
  diary_file
  data_file
  target_file
  epsilon
  options

  load(data_file); % X_train y_train X_test y_test
  options_old = options;
  load(target_file); % thetas test_errors train_losses reps quantiles options
  options = options_old; % hack because target_file has options
  num_targets = length(test_errors); % number of target parameters
  best_error = 0.0;
  options
  if options.save_realized
    theta_realized = cell(num_targets,1);
    bias_realized = cell(num_targets,1);
    err_realized = zeros(num_targets,1);
  end
  if options.save_all
    attack_dir = sprintf('%s_attacks', diary_file(1:end-4));
    fprintf(1, 'saving attacks to %s\n', attack_dir);
    mkdir(attack_dir);
    assert(length(options.rho_max) == 1);
    assert(length(options.att_rep) == 1);
  end
  switch options.ordering
    case 'default'
      % just go through the target parameters in order
      i_order = 1:num_targets;
      i_order
    case 'sqrt'
      % fancier ordering that attempts to explore the space more efficiently
      skip = round(sqrt(num_targets));
      initial = 1:skip:num_targets;
      rest = setdiff(1:num_targets, initial);
      rest = rest(length(rest):-1:1);
      i_order = [initial  rest];
      i_order
    otherwise
      assert(false);
  end

  for i=i_order
    % Decide whether to skip this parameter (because we've already run it before, or it doesn't look promising)
    tic;
    aux_time = 0;
    if options.save_all
      save_loc = sprintf('%s/%d.mat', attack_dir, i);
      if exist(fullfile(cd, save_loc), 'file')
        fprintf(1, '%s already exists, skipping...\n', save_loc);
        continue;
      end
    end
    theta_tar = thetas{i};
    bias_tar = biases{i};
    error_tar = test_errors(i);
    if isfield(options, 'max_train_loss') && train_losses(i) > options.max_train_loss
      fprintf(1, 'train loss (%.4f) higher than maximum (%.4f), skipping...\n', train_losses(i), options.max_train_loss);
      continue;
    end
    if options.skip && error_tar < best_error
      fprintf(1, 'target error (%.4f) less than best error (%.4f), skipping...\n', error_tar, best_error);
      continue;
    end
    diary off; diary on; % hack to flush output

    aux_time = aux_time + toc;

    % Generate attack
    tic;
    if options.use_train
      if isfield(options, 'really_use_train') && options.really_use_train
        % this will make it harder to compute diagnostics, but makes sure that generateAttackTar never even sees the test set
        [X_attack, y_attack, ~, ~, base_err] = generateAttackTar(X_train, y_train, X_train, y_train, X_train, y_train, theta_tar, bias_tar, epsilon, options);
      else
        [X_attack, y_attack, ~, ~, base_err] = generateAttackTar(X_train, y_train, X_train, y_train, X_test, y_test, theta_tar, bias_tar, epsilon, options);
      end
    else
      [X_attack, y_attack, ~, ~, base_err] = generateAttackTar(X_train, y_train, X_test, y_test, X_test, y_test, theta_tar, bias_tar, epsilon, options);
    end
    attack_time = toc;

    % Evaluate result
    tic;
    scores = struct();
    for tr=1:options.evaluation_trials
      scoresCur = evaluateDefenses(X_train, y_train, X_attack, y_attack, X_test, y_test, options);
      allFields = fields(scoresCur);
      for j=1:length(allFields)
        curVal = getfield(scoresCur, allFields{j});
        if isfield(scores, allFields{j})
          oldVal = getfield(scores, allFields{j});
          scores= setfield(scores, allFields{j}, [oldVal  curVal]);
        else
          scores = setfield(scores, allFields{j}, [curVal]);
        end
      end
    end
    eval_time = toc;

    % Print stats and save attack data
    fprintf(1, 'Done with attack (theta_num = %d, error_tar = %.4f, rho_max = %f, att_rep = %d)\n', i, error_tar, options.rho_max, options.att_rep);
    if options.evaluation_trials > 0
      fprintf(1, 'Printing scores...\n');
      prettyPrint(scores);
    end
    if options.save_realized
      % Save the actual parameters we get from training on the poisoned data
      [theta_realized_cur,~,~,~,err_realized_cur,bias_realized_cur,~] = train([X_train;X_attack], [y_train;y_attack], X_test, y_test, options);
      theta_realized{i} = theta_realized_cur;
      bias_realized{i} = bias_realized_cur;
      err_realized(i) = err_realized_cur;
      save(sprintf('%s.realized.mat', diary_file), 'theta_realized', 'bias_realized', 'err_realized');
    end
    if options.save_all
      fprintf(1, 'done with attack %d/%d: attack_time = %f seconds, eval_time = %f seconds\n', i, num_targets, attack_time, eval_time);
      fprintf(1, 'saving output to %s\n', save_loc);
      elapsed = systime - start;
      save(save_loc, 'X_attack', 'y_attack', 'options', 'scores', 'attack_time', 'eval_time', 'aux_time', 'elapsed');
    end
    if options.evaluation_trials > 0
      cur_error = options.aggregator(scores);
      if cur_error > best_error
        X_attack_best = X_attack;
        y_attack_best = y_attack;
        best_error = cur_error;
      end
      fprintf(1, 'best error so far: %.4f\n', best_error);
    end
  end
  diary off;
  if ~options.save_all
    save(sprintf('%s.mat', diary_file), 'X_attack_best', 'y_attack_best', 'best_error');
  else
    fprintf(1, 'aggregating scores...\n');
    optionsGlobal = options;
    optionsGlobal.result_dir = attack_dir;
    optionsGlobal.datafile = data_file;
    aggregateScores(attack_dir, optionsGlobal);
  end
  fprintf(1, 'total elapsed time: %f\n', systime - start);
end

% Fill in default options if they aren't already set
function options = processOptionsGenerateAttackAll(options)
  if ~isfield(options, 'p')
    options.p = 0.30;
  end
  if ~isfield(options, 'burn_frac')
    options.burn_frac = 1.0;
  end
  if ~isfield(options, 'decay')
    options.decay = 0.0;
  end
  if ~isfield(options, 'method')
    options.method = 'adagrad';
  end
  if ~isfield(options, 'save_realized')
    options.save_realized = 0;
  end
  if ~isfield(options, 'evaluation_trials')
    options.evaluation_trials = 7;
  end
  if ~isfield(options, 'rho_max')
    options.rho_max = 0.25;
  end
  if ~isfield(options, 'att_rep')
    options.att_rep = 2;
  end
  if ~isfield(options, 'skip')
    options.skip = 1;
  end
  if ~isfield(options, 'ordering')
    options.ordering = 'default';
  end
  assert(isfield(options, 'aggregator'));
  assert(isfield(options, 'guarder'));
end

function t = systime()
  c = clock;
  t = c(6) + 60 * c(5) + 60 * 60 * c(4) + 60 * 60 * 24 * c(3);
end
