function y_out = fun_local_i( u )
% u= [EV_i; G_i], 48*1

global eps_EV eps_G 
global B
global dim_A

global Lambda_old

mu = Lambda_old(1:dim_A*4);
nu_G = Lambda_old(dim_A*4+1:dim_A*5);

% disp('size(u)')
% size(u)

y_out = u(1:24)'*B*mu + nu_G'*u(25:48) + 0.5*eps_EV*u(1:24)'*u(1:24) + 0.5*eps_G*u(25:48)'*u(25:48);
%y_out = u(1:24)*B*mu + nu_G'*u(25:48)' + 0.5*eps_EV*u(1:24)*u(1:24)' + 0.5*eps_G*u(25:48)*u(25:48)';


end

