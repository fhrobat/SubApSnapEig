function indices = lu_sampling(V, s)
    [~,~,indices] = lu(V,'vector'); 
    if s > size(V,2)
        indices = indices(1:size(V,2));
    else
        indices = indices(1:s);
    end