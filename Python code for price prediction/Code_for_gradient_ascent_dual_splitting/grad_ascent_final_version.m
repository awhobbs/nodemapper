clear
clc
format long
%%% apply gradient ascent method to obtain optimal value for dual variables

my_flag = 1; 
% my_flag = 1 for normal gradient ascent

load('input.mat')
G_i_low_const = -10.0; % kWh
G_i_up_const = +10.0; % kWh


% define global parameters for the problem
global eps_mu eps_EV eps_G 
global lambda Sigma_p p_tilde
global dim_A A B
global N
global c
global Lambda_old Lambda_new


N = 100; % number of houses
eps_mu = 1;
eps_EV = 1;
eps_G = 1;
c = [e_inf_agg_hat'; -e_sup_agg_hat'; EV_inf_agg_hat'; -EV_sup_agg_hat'];
lambda = 1.0;
Sigma_p = p_cov; 
p_tilde = p_e;
dt = 1;
dim_A = 24;
A = dt*tril(ones(dim_A,dim_A));
B = [-A',A',-eye(dim_A),eye(dim_A)];

%%%%%%%%%%%%%%%%%%%%%%%%
% compute strongly concave constant m and M and M_bar
[~, D_eig] = eig(Sigma_p);
D_vector = diag(D_eig);
[lambda_max, ~] = max(D_vector);
[lambda_min, ~] = min(D_vector);
m = min(1.0/2.0/lambda/lambda_max,eps_mu);
M = max(1.0/2.0/lambda/lambda_min,eps_mu);
eps_u = min(eps_EV, eps_G);
theta_1 = blkdiag(B,eye(24));
M_bar = M + N*(norm(theta_1))^2/eps_u;

% % compute the number of iterations needed for convergence to certain error


error_size = 1.0; % initialize error_size
tol = 1e-1;


% define initial guess for dual variables
Lambda_0 = 0.1*ones(120,1);
Lambda_0(1:24*4) = 0.000001;
% Lambda needs to be a column vector
Lambda_old = Lambda_0; % Lambda_old is used to store the old solution
Lambda_old_old = Lambda_old;
%Lambda_new = Lambda_0; % Lambda_new is used to store the new solution

u_store = zeros(N,48); % used to store solution for all N individual local optimization problem
fun_local_store = zeros(N,1); % used to store local optimal value

step_size = 1/M_bar; % step size for gradient ascent
disp('step_size = ')
disp(step_size)

iter = 1;
max_iter = 200;


% store function value of g(Lambda) for the three key steps
g_Lambda_last = 0.0;
g_Lambda_second_last = 0.0;
g_Lambda_first = 0.0;

error_store =[];
while (error_size>= tol) && (iter <= max_iter)
    
    disp(strcat('iter = ', num2str(iter)))
    
    % split the input vector Lambda = [mu;nu_G], Lambda is a column vector
    mu = Lambda_old(1:dim_A*4);
    nu_G = Lambda_old(dim_A*4+1:dim_A*5);
    
    % find local optimal solution u_i_k = [EV_i; G_i], the optimal solution for the i th local problem at the k th iteration
    grad_g = [-eps_mu*mu+c; -1.0/2.0/lambda*Sigma_p\(nu_G-p_tilde)]; % initial component of grad_g
    
    for i = 1:N
        ui_0 = zeros(48,1);
        
        % u(1:24) represent EV_i
        % u(25:48) represent G_i
        %fun_local_i = @(u) u(1:24)'*B*mu + nu_G'*u(25:48) + 0.5*eps_EV*u(1:24)'*u(1:24) + 0.5*eps_G*u(25:48)'*u(25*48);
        A_ineq_i = [A, zeros(24,24); -A, zeros(24,24); eye(24),-eye(24)];
        b_ineq_i = [e_sup_hour_avg(i,:)';-1*e_inf_hour_avg(i,:)';S(i,:)' - L(i,:)'];
        Aeq_i = [];
        beq_i = [];
        EV_i_lb = EV_inf_hour_avg(i,:)';
        EV_i_ub = EV_sup_hour_avg(i,:)';
        G_i_lb = G_i_low_const*ones(24,1);
        G_i_ub = G_i_up_const*ones(24,1);
        lb_i = [EV_i_lb;G_i_lb];
        ub_i = [EV_i_ub;G_i_ub];
        options = optimset('Display', 'off') ;
        [ui,fval_local_i] = fmincon(@fun_local_i,ui_0,A_ineq_i,b_ineq_i,Aeq_i,beq_i,lb_i,ub_i,[],options);
        
        % assgin local solutions to the storage matrix
        u_store(i,:) = ui;
        fun_local_store(i) = fval_local_i;
        
        % assemble grad_g
        grad_g = grad_g + [B'*ui(1:24);ui(25:48)];
    end
    
    
    % compute grad_g
    % grad_g = [-eps_mu*mu+c; -1.0/2.0/lambda*Sigma_p\(nu_G-p_tilde)]
    
    % update value of dual variables using gradient ascent (i.e. compute Lambda_new)
    Lambda_new_temp = Lambda_old + step_size * grad_g;
    
    % modify Lambda_new_temp based on projected gradient ascent method
    mu_new_temp = Lambda_new_temp(1:dim_A*4);
    modify_entry = [sign(subplus(mu_new_temp));ones(24,1)];
    
    %Lambda_new = Lambda_old + (H_inv * grad_g).*modify_entry;
    Lambda_new = Lambda_old + step_size * (grad_g.*modify_entry);
    
    
    % maybe when I compute the error_size, I need to compute mu and nu_G differently
    error_size_1 = norm(Lambda_new(1:2*24) - Lambda_old(1:2*24))/norm(Lambda_old(1:2*24))
    error_size_2 = norm(Lambda_new(2*24+1:4*24) - Lambda_old(2*24+1:4*24))/norm(Lambda_old(2*24+1:4*24))
    error_size_3 = norm(Lambda_new(4*24+1:5*24) - Lambda_old(4*24+1:5*24))/norm(Lambda_old(4*24+1:5*24))
    
    %error_size = norm(Lambda_new - Lambda_old)/norm(Lambda_old);
    error_size = error_size_1 + error_size_2 + error_size_3;

    if iter == 1
        % record the function value of g(Lambda) for the first step
        g_Lambda_first = g(Lambda_old, fun_local_store);
    end
    
    if iter == 199
        % record the function value of g(Lambda) for the second last step
        g_Lambda_second_last = g(Lambda_old, fun_local_store);
    end
    
    
    if iter == 200
        % record the function value of g(Lambda) for the last step
        g_Lambda_last = g(Lambda_old, fun_local_store);
    end
    
    Lambda_old = Lambda_new;
    
    disp('error_size = ');
    disp(error_size);
    
    error_store = [error_store, error_size];
    
    iter = iter + 1;
    
end



