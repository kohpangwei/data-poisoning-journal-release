function [thetas, biases, train_losses, test_errors, quantiles, reps] = pruneTheta(thetas, biases, train_losses, test_errors, quantiles, reps)
  [~,iisort] = sort(test_errors);
  iisort = flip(iisort);
  iisort_pruned = [];
  min_train_loss = 1e9;
  for ii=iisort'
    if train_losses(ii) < min_train_loss
      iisort_pruned = [iisort_pruned; ii];
      min_train_loss = train_losses(ii);
    end
  end
  iisort = iisort_pruned;
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

