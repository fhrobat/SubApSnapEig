function [indices,probs] = samp(V, s, sampling)
    if sampling == "leverage"
        [indices, probs] = leverage_score_sampling(V, s);
    elseif sampling == "lu"
        indices = lu_sampling(V, s);
        probs = [];
    else
        error('select valid sampling method')
    end
end