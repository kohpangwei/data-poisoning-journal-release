% generates the min-max attack for a given set of target parameters (theta_fake, bias_fake)
function [X_attack, y_attack, X_best, y_best, base_err]=generateAttackTar(X_train, y_train, X_dev, y_dev, X_test, y_test, theta_fake, bias_fake, epsilon, options) 

    n = size(X_train,1);
    d = size(X_train,2);
    options = processOptionsGenerateAttackTar(options, n);

    yalmip('clear'); % for better efficiency
    
    is_binary = length(unique(y_train)) <= 2;
    if ~is_binary
        num_classes = max(y_train);
        num_classes_adjust = num_classes;
        y_list = (1:num_classes)';
    else
        num_classes = 2;
        % num_classes_adjust affects the dimension of theta;
        % if we are doing binary classification theta has d parameters, for k > 2 classes theta has d*k parameters
        num_classes_adjust = 1; 
        y_list = [1;-1];
    end

    % generates parameters for data sanitization
    % the data sanitization schemes generally throw away points that are far away from some "mean" of the data
    % this subroutine computes the relevant means (mu_guard) and the threshold for throwing points away (rho_guard)
    [mu_guard, rho_guard, num_guard, rho_guard_slab] = guard(theta_fake, X_train, y_train, num_classes, d, y_list, options);
    
    discrete = all(all(X_train == floor(X_train)));
    n_att = round(epsilon*n);
    % run min-max algorithm for n_burn "burn-in" rounds initially before starting to collect attack points
    n_burn = round(options.burn_frac*n_att);
    if discrete
      % discrete problems also tend to be high-dimensional with many zeros, so we use a sparse representation
      X_attack = sparse(n_burn + n_att, d);
    else
      X_attack = zeros(n_burn + n_att, d);
    end
    y_attack = zeros(n_burn + n_att, 1);
    X_best = [];
    y_best = [];
    best_err = 0.0;

    [theta,~,losses,theta2,base_err] = train(X_train, y_train, X_test, y_test, options);
    fprintf(1, 'theta norm: %.4f, theta2 norm: %.4f\n', norm(theta,2), norm(theta2,2));
    fprintf(1, 'base error: %.4f\n', base_err);
    
    bias = zeros(1, num_classes_adjust);
    bias2 = 1 * ones(1, num_classes_adjust);

    % x is the optimization variable for choosing a single attack point
    x = sdpvar(d,1);
    % we will constrain x to only take on values within the range of what occurred in either the training or dev set
    % note in our experiments X_dev is actually equal to either X_train or X_test depending on whether use_train == 1
    if discrete && options.LP
      % special case: if a discrete variable only ever takes on the value 0, we also allow it to take on the value 1
      xmax = max(max(max(X_train, [], 1)', max(X_dev, [], 1)'), ones(d,1));
      xmin = min(min(X_train, [], 1)', min(X_dev, [], 1)');
      if isfield(options, 'nneg_only')
        fprintf(1, 'only using nneg constraints\n');
        Constraint0 = [x >= 0];
      else
        Constraint0 = [x >= xmin; x <= xmax];
      end
      kmax = max(xmax);
      % env is a piecewise-linear upper bound on x^2
      % it is also equal to the expected value of x^2 under our randomized rounding scheme
      % this will be useful later because we will have constraints involving x^2 that we would like to represent as an LP
      env = sdpvar(d,1);
      for k=1:kmax
          active = (xmin <= k & k <= xmax);
          Constraint0 = [Constraint0; env(active) >= x(active) * (2*k-1) - k*(k-1)];
      end
    else
      xmax = max(max(X_train, [], 1)', max(X_dev, [], 1)');
      xmin = min(min(X_train, [], 1)', min(X_dev, [], 1)');
      Constraint0 = [x >= xmin; x <= xmax];
    end

    best_bound = 1e9;

    X_curs = [];
    y_curs = [];
    i = 0;
    while i < (n_att+n_burn)
        objs = zeros(num_classes,1);
        xs = zeros(num_classes,d);
        % iterate over all possible classes that we could choose for the attack point
        for j=1:num_classes
            y = y_list(j);
            for g=1:num_guard
              m = mu_guard{g}(j,:);
              r = rho_guard{g}(j);
              % constrain data points to get past data sanitization
              % recall that env is an LP upper bound on x^2 that we use when the data is discrete
              if discrete && options.LP
                Constraint = [Constraint0; sum(env) - 2*m*x + norm(m,2)^2 <= r^2];
              else
                Constraint = [Constraint0; norm(x,2)^2 - 2*m*x + norm(m,2)^2 <= r^2];
              end
              % if doing binary classification we might also use a "slab" defense for data sanitization
              % that has a different functional form
              if options.slab
                assert(is_binary);
                m2 = mu_guard{g}(3-j,:);
                v_slab = m2 - m;
                r_slab = rho_guard_slab{g}(j);
                Constraint = [Constraint; v_slab * (x - m') <= r_slab];
                Constraint = [Constraint; -v_slab * (x - m') <= r_slab];
              end
            end
            % next constrain attack point to have small loss under the target parameter theta_fake
            if is_binary
              % in the binary case constraining the loss is straightforward
              if ~isempty(theta_fake)
                Constraint = [Constraint; 1 - y * (theta_fake'*x + bias_fake) <= options.rho_max];
              end
              % objective is to make the loss under the current parameter theta as large as possible
              Objective = 1-y*(theta'*x+bias);
            else
              % in the non-binary case this is trickier; we have to look at all num_classes-1 potential wrong classes
              for j2=1:num_classes
                if j2~=j
                  if ~isempty(theta_fake)
                    Constraint = [Constraint; 1-((theta_fake(:,j)-theta_fake(:,j2))'*x + (bias_fake(j) - bias_fake(j2))) <= options.rho_max];
                  end
                end
              end
              % in the non-binary case, maximizing the loss under theta is non-convex; 
              % instead pick a random target class j_tar and maximize the loss for that target class
              while true
                j_tar = randint(1,1,num_classes)+1;
                if j_tar ~= j
                  break;
                end
              end
              Objective = 1 - ((theta(:,j)-theta(:,j_tar))'*x + (bias(j) - bias(j_tar)));
            end
            % send the optimization to gurobi
            optimize(Constraint, -Objective, sdpsettings('verbose', 0, 'solver', 'gurobi', 'gurobi.NumericFocus', 1, 'cachesolvers', 1));
            objs(j) = double(Objective);
            xs(j,:) = double(x);
        end
        % print what objective value (maximum loss) we can achieve for each possible class
        fprintf(1, 'iter %d objs:', i);
        for j=1:num_classes
          fprintf(1, ' %.4f', objs(j));
        end
        fprintf(1, '\n');
        % choose the class that achieved the best objective value, and use that one for the attack
        [~,j_cur] = max(objs);
        % repeat the attack point att_rep times
        for r=1:options.att_rep
          % for discrete data, round x independently for each repetition
          if discrete
            [x_cur,prob_eq] = randRound(xs(j_cur,:)');
          else
            x_cur = xs(j_cur,:)';
          end
          y_cur = y_list(j_cur);

          % add x to attack set
          i = i+1;
          X_curs = [X_curs; x_cur'];
          y_curs = [y_curs; y_cur];
          X_attack(i,:) = x_cur;
          y_attack(i,1) = y_cur;
        end

        % gradient update for min-max objective
        [g_c, L_c, dbias_c, err_c] = nabla_Loss(X_train, y_train, theta, bias, options);
        [g_p, L_p, dbias_p, err_p] = nabla_Loss(x_cur', y_cur, theta, bias, options);
        acc_c = 1-err_c;
        acc_p = 1-err_p;

        % if theta_fake = [], then we are running the attack from the NIPS submission, which 
        % gives an approximate upper bound on the maximum loss achievably by any attack
        if isempty(theta_fake)
          bound = L_c + epsilon * L_p + 0.5 * options.decay * norm(theta, 'fro')^2;
          best_bound = min(bound, best_bound);
          fprintf(1, 'bound: %.4f | best_bound: %.4f\n', bound, best_bound);
        end
        fprintf(1, 'loss: %.4f (clean) | %.4f (poisoned) | %.4f (all)\n', L_c, L_p, L_c + epsilon * L_p);
        fprintf(1, ' acc: %.4f (clean) | %.4f (poisoned)\n', acc_c, acc_p);
        fprintf(1, 'norm of params: %.4f | bias: ', norm(theta,'fro')); disp(bias);
        % combine gradients of different terms and perform update
        g = g_c + epsilon * g_p;
        g = g + options.decay * theta;
        dbias = dbias_c + epsilon * dbias_p;
        theta2 = theta2 + g.^2;
        theta = theta - options.eta_mm * g ./ sqrt(theta2);
        bias2 = bias2 + dbias.^2;
        bias = bias - options.eta_mm * dbias ./ sqrt(bias2);
        
        % every 25 iterations we will train against the poisoned data so far to see how we're doing
        if mod(i,25) < options.att_rep && options.evaluation_trials > 0
            % ignore the n_burn / (n_att + n_burn) fraction of points at the beginning
            i1 = ceil(i * n_burn / (n_att + n_burn));
            i2 = i;
            % since the number of points we've generated so far is less than the total number of points allowed (n_att), 
            % we will repeat the points to get to n_att points in total
            X_pois = repmat(X_attack(i1:i2,:), [ceil(n_att/(i2-i1+1)) 1]);
            y_pois = repmat(y_attack(i1:i2,1), [ceil(n_att/(i2-i1+1)) 1]);
            X_pois = X_pois(1:n_att,:);
            y_pois = y_pois(1:n_att,1);
            [theta_cur,~,loss_cur,theta2_cur,err_test,bias_cur,~] = train([X_train;X_pois], [y_train;y_pois], X_test, y_test, options);
            [~, loss_train, ~, err_train] = nabla_Loss(X_train, y_train, theta_cur, bias_cur, options);
            [~, loss_dev,   ~, err_dev]   = nabla_Loss(X_dev,   y_dev,   theta_cur, bias_cur, options);
            [~, loss_test,  ~, err_test]  = nabla_Loss(X_test,  y_test,  theta_cur, bias_cur, options);
            [~, loss_pois,  ~, err_pois]  = nabla_Loss(X_pois,  y_pois,  theta_cur, bias_cur, options);
            fprintf(1, 'theta norm: %.4f, theta2 norm: %.4f\n', norm(theta_cur,2), norm(theta2_cur,2));
            fprintf(1, 'loss: %.4f (all), %.4f (pois), %.4f (train), %.4f (dev), %.4f (test)\n', mean(loss_cur), loss_pois, loss_train, loss_dev, loss_test);
            fprintf(1, 'err: %.4f (pois), %.4f (train), %.4f (dev), %.4f (test)\n', err_pois, err_train, err_dev, err_test);
            % save the results in case one of the intermediate parameters is better than our final params
            if err_dev > best_err
              best_err = err_dev;
              X_best = X_pois;
              y_best = y_pois;
            end
            fprintf(1, ':: err: %.4f, best_err: %.4f\n', err_dev, best_err);
        end
        
    end
    X_attack = X_attack(n_burn + (1:n_att), :);
    y_attack = y_attack(n_burn + (1:n_att), 1);
end

function options = processOptionsGenerateAttackTar(options, n)
  assert(isfield(options, 'p'));
  assert(isfield(options, 'burn_frac'));
  assert(isfield(options, 'rho_max'));
  assert(isfield(options, 'att_rep'));
  assert(isfield(options, 'decay'));
  assert(isfield(options, 'guarder'));
  if ~isfield(options, 'eta')
    options.eta = 0.02;
  end
  if ~isfield(options, 'eta_mm')
    options.eta_mm = 0.03;
  end
  if ~isfield(options, 'batch_size')
    options.batch_size = min(100, ceil(0.005 * n));
  end
  if ~isfield(options, 'LP')
    options.LP = 1;
  end
end


%% Given a guard function, filter points appropriately
function [mu_guard, rho_guard, num_guard, rho_guard_slab] = guard(theta, X_train, y_train, num_classes, d, y_list, options)
  is_binary = length(unique(y_train)) <= 2;
  theta_guard = options.guarder(theta);
  num_guard = length(theta_guard);
  mu_guard = cell(num_guard,1);
  rho_guard = cell(num_guard,1);
  rho_guard_slab = cell(num_guard,1);
  for g=1:num_guard
    g
    mu = zeros(num_classes,d);
    rho = zeros(num_classes,1);
    Ls = Losses(X_train, y_train, theta_guard{g});
    active = Ls > 0;
    for j=1:num_classes
        match_y = (y_train==y_list(j));
        cur_active = active & match_y;
        p_active = sum(cur_active) / sum(match_y);
        data = X_train(cur_active,:);
        m = mean(data, 1);
        norms = sqrt(sum(data.^2, 2) - 2 * data * m' + norm(m,2)^2);
        mu(j,:) = m;
        rho(j) = quantile(norms, max(0.05, 1-options.p/p_active));
        fprintf(1, 'quantile for class %d: %.4f (p=%.4f, p_active=%.4f)\n', j, max(0.05, 1-options.p/p_active), options.p, p_active);
    end
    rho
    mu_guard{g} = mu;
    rho_guard{g} = rho;
    if is_binary % add in slab info
      rho_slab = zeros(num_classes,1);
      for j=1:num_classes
        m = mu(j,:);
        m2 = mu(3-j,:);
        v_slab = m - m2;

        match_y = (y_train==y_list(j));
        cur_active = active & match_y;
        p_active = sum(cur_active) / sum(match_y);
        data = X_train(cur_active,:);

        projs = abs(data * v_slab' - m * v_slab');
        fprintf(1, 'size of projs: %d\n', size(projs));
        rho_slab(j) = quantile(projs, max(0.05, 1-options.p/p_active));
      end
      rho_guard_slab{g} = rho_slab;
    end
  end
end


%% Minor helper functions
function Ls = Losses(X, y, theta)
  if size(theta,2) == 1
    Ls = max(0, 1 - (y .* (X * theta)));
  else
    Ls = compute_losses(X, y, theta, size(theta, 2));
  end
end


function losses=compute_losses(X, y, theta, num_classes)
    [b_indices, m_indices, b_scores, m_scores, ym] = process(X*theta, y, num_classes);
    margins = b_scores-m_scores;
    losses = max(0, 1 - margins);
end
