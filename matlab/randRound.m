function [xr, prob_eq] = randRound(x)
    d = length(x);
    assert(d == numel(x));
    x_round = round(x);
    x_rem = x - x_round;
    x_rem_sparse = sign(x_rem) .* (rand(d,1) < abs(x_rem));
    xr = x_round + x_rem_sparse;
    prob_eq = prod(abs(x_rem).^2 + (1-abs(x_rem)).^2);
end
