function enronAttackScript(slab)
  options = struct();
  options.LP = 1;
  options.timeit = 0;
  options.nips = 0;
  options.slab = slab;
  att_rep = 2;
  grad = 0;

  for epsilon=[0.03] %[0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    options.use_train = 0;
    generateAttackFnc('enron', epsilon, att_rep, grad, options);

    options.use_train = 1;
    generateAttackFnc('enron', epsilon, att_rep, grad, options);
  end
end
