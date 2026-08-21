function A = exp_full(mu, L, N)

if nargin < 3
    error('Usage: A = Afull(mu, L, N)');
end

x = linspace(-L, L, N);
xm = 0.5 * (x(1:end-1) + x(2:end)); 
a_mid = exp(mu * xm); 
n = N - 2; % Number of interior nodes

am_main = a_mid(1:end-1)'; % a_{i-1/2}
ap_main = a_mid(2:end)';   % a_{i+1/2}
main_diag = -(ap_main + am_main); 

a_int = a_mid(2:end-1)'; % a_{3/2}, a_{5/2}, ..., a_{n-1/2}

sub_diag   = [a_int; 0]; 
super_diag = [0; a_int]; 

A = spdiags([sub_diag, main_diag, super_diag], [-1, 0, 1], n, n);

end