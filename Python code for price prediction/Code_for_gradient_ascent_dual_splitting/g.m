function y_out = g( Lambda, fun_local_store)
% g(Lambda) computes the function value for the global maximization problem.
% fun_local_store is an array (N*1) storing all individual local optimal function values

global eps_mu  
global lambda Sigma_p p_tilde
global dim_A
global c


% split the input vector Lambda = [mu;nu_G]
mu = Lambda(1:dim_A*4);
nu_G = Lambda(dim_A*4+1:dim_A*5);

% construct global part of the function g(Lambda)
phi_0 = -1.0/4.0/lambda*(nu_G - p_tilde)'*(Sigma_p\(nu_G - p_tilde)) + c'*mu- 1.0/2.0*eps_mu*(mu'*mu);

% construct local part of the function g(Lambda)
sum_phi_i = sum(fun_local_store);

y_out = phi_0 + sum_phi_i;

end

