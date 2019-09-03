% Generates a set of target parameters via the following procedure:
%   For each q in options.quantile_tape and r in options.rep_tape:
%     1. Train a model theta on (X_train, y_train)
%     2. Take the qth quantile of points in X_dev that would be 
%        *most incorrectly* classified if we were to switch the label to a different label. 
%        Call the resulting dataset (with switched labels) X_tar, y_tar
%     3. Train a new model on (X_train, y_train) plus r copies of (X_tar, y_tar)
%     4. Save the resulting parameter theta_fake
%   If options.prune is set to 1:
%     Then additionally remove all values of theta_fake that 
%     are not on the Pareto boundary of (train_loss, test_error)
function [thetas, biases, train_losses, test_errors, quantiles, reps, options] = generateTheta(X_train, y_train, X_dev, y_dev, X_test, y_test, options)

    % Process options and set dataset parameters
    n = size(X_train,1);
    d = size(X_train,2);
    n_dev = size(X_dev, 1);
    options = processOptionsGenerateTheta(options, n);
    options

    thetas = {};
    biases = {};
    train_losses = [];
    test_errors = [];
    quantiles = [];
    reps = [];

    
    is_binary = length(unique(y_train)) <= 2;
    if ~is_binary
        num_classes = max(y_train);
        y_list = (1:num_classes)';
    else
        num_classes = 2;
        y_list = [1;-1];
    end

    % Train initial model
    [theta,~,losses,theta2,~,bias,~] = train(X_train, y_train, X_test, y_test, options);
    fprintf(1, 'theta norm: %.4f, theta2 norm: %.4f\n', norm(theta,2), norm(theta2,2));
    fprintf(1, 'bias: '); disp(bias);
    [~,losses_test,~,~] = nabla_Loss(X_test, y_test, theta, bias, options);
    fprintf(1, 'average loss: %.4f (train), %.4f (test)\n', mean(losses), losses_test);
    
    
    % Iterate over possible (q, r) values
    for loss_quantile = options.quantile_tape
      for tar_rep = options.rep_tape
        fprintf(1, '>>> Generating attack with params QUANTILE=%f, REP=%d\n', loss_quantile, tar_rep);
        % Compute worst-case label flip and corresponding margin
        if is_binary
          ym = -y_dev;
          margins = y_dev.* (X_dev * theta + bias);
        else
          [b_indices, m_indices, b_scores, m_scores, ym] = process(X_dev*theta + repmat(bias, [n_dev 1]), y_dev, num_classes);
          margins = b_scores-m_scores;
        end

        % Keep points in dev set whose margin is within the qth quantile
        % Save result as (X_tar, y_tar)
        margin_thresh = quantile(margins, loss_quantile);

        fprintf(1, '\tfraction of points used: %.4f\n', mean(margins<margin_thresh));
        X_tar = [];
        y_tar = [];
        y_orig = [];
        for j=1:num_classes
          % Use below definition of margin_thresh to take the qth quantile class-by-class rather than overall.
          %margin_thresh = quantile(margins(y_dev == y_list(j)), loss_quantile);
          active_cur = y_dev == y_list(j) & margins<margin_thresh;
          X_tar_cur = X_dev(active_cur,:);
          y_tar_cur = ym(active_cur, 1);
          y_orig_cur = y_dev(active_cur, 1);
          X_tar = [X_tar; X_tar_cur];
          y_tar = [y_tar; y_tar_cur];
          y_orig = [y_orig; y_orig_cur];
        end

        n_tar = size(X_tar,1);
        

        % Train on new dataset consisting of (X_train, y_train) + r repetitions of (X_tar, y_tar)
        [theta_fake,~,losses,theta2_fake,err_test,bias_fake] = train([X_train;repmat(X_tar, [tar_rep 1])],[y_train;repmat(y_tar, [tar_rep 1])], X_test, y_test, options);
        [~,train_loss,~,~] = nabla_Loss(X_train, y_train, theta_fake, bias_fake, options);
        fprintf(1, 'theta norm: %.4f, theta2 norm: %.4f\n', norm(theta_fake,2), norm(theta2_fake,2));
        fprintf(1, '\taverage loss for fake params: %.4f\n', train_loss);
        fprintf(1, '\ttest error for fake params: %.4f\n', err_test);

        % Append results to our list of target parameters
        thetas = [thetas; theta_fake];
        biases = [biases; bias_fake];
        train_losses = [train_losses; train_loss];
        test_errors = [test_errors; err_test];
        quantiles = [quantiles; loss_quantile];
        reps = [reps; tar_rep];
      end
    end

    % Prune away target parameters that are not on the Pareto boundary of (train_loss, test_error)
    [~,iisort] = sort(test_errors);
    iisort = flip(iisort);
    if options.prune
      iisort_pruned = [];
      min_train_loss = 1e9;
      for ii=iisort'
        if train_losses(ii) < min_train_loss
          iisort_pruned = [iisort_pruned; ii];
          min_train_loss = train_losses(ii);
        end
      end
      iisort = iisort_pruned;
    end
    thetas = thetas(iisort);
    biases = biases(iisort);
    train_losses = train_losses(iisort);
    test_errors = test_errors(iisort);
    quantiles = quantiles(iisort);
    reps = reps(iisort);
end

function y=flip(x)
  y=x(end:-1:1);
end

% Fill in options with default if they are not already set
% It is recommended to at least set quantile_tape and rep_tape as these are likely dataset-specific
function options = processOptionsGenerateTheta(options, n)
  if ~isfield(options, 'decay')
    options.decay = 0.0;
  end
  if ~isfield(options, 'method')
    options.method = 'adagrad';
  end
  if ~isfield(options, 'batch_size')
    options.batch_size = min(100, ceil(0.005 * n));
  end
  if ~isfield(options, 'quantile_tape')
    options.quantile_tape = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35];
  end
  if ~isfield(options, 'rep_tape')
    options.rep_tape = [1, 2, 3, 5, 8, 12, 18, 25];
  end
  if ~isfield(options, 'prune')
    options.prune = 1;
  end
end
