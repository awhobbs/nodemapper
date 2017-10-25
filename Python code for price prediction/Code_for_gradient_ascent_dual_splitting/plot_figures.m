my_flag = 1;
%my_flag = 1; normal gradient ascent
%my_flag = 2; accelerated gradient ascent
%my_flag = 3; newton

t = 1:24;
if my_flag == 1
    U_aggregate = ones(1,100)*u_store_preconditioned;
    L_aggregate = ones(1,100)*L;
    S_aggregate = ones(1,100)*S;

    figure
    plot(t,U_aggregate(25:48),'b',t, U_aggregate(1:24),'c',t,L_aggregate,'r',t,S_aggregate,'g')
    axis([1 24 -300 400])
    title('Gradient ascent method')
    legend('\Sigma G','\Sigma EV','\Sigma L','\Sigma S')
    xlabel('Time(hour)')
    ylabel('Power(kWh)')
    
%elseif my_flag == 2
    U_aggregate = ones(1,100)*u_store_AGA;
    L_aggregate = ones(1,100)*L;
    S_aggregate = ones(1,100)*S;

    figure
    plot(t,U_aggregate(25:48),'b',t, U_aggregate(1:24),'c',t,L_aggregate,'r',t,S_aggregate,'g')
    axis([1 24 -300 400])
    title('Accelerated gradient ascent method')
    legend('\Sigma G','\Sigma EV','\Sigma L','\Sigma S')
    xlabel('Time(hour)')
    ylabel('Power(kWh)')
    
%elseif my_flag == 3
    U_aggregate = ones(1,100)*u_store_newton;
    L_aggregate = ones(1,100)*L;
    S_aggregate = ones(1,100)*S;

    figure
    plot(t,U_aggregate(25:48),'b',t, U_aggregate(1:24),'c',t,L_aggregate,'r',t,S_aggregate,'g')
    axis([1 24 -300 400])
    title('Newton''s method')
    legend('\Sigma G','\Sigma EV','\Sigma L','\Sigma S')
    xlabel('Time(hour)')
    ylabel('Power(kWh)')
end

figure
iter = 1:200;
semilogy(iter, error_store_preconditioned, iter, error_store_AGA, iter, error_store_newton)
%yticks(logspace(-5,10,2))
set(gca,'YTick',logspace(-5,9,15))
%set(gca,'YTickLabel',['10^-5';'10^-4';'10^-3';'10^-2';'10^-1';'10^{ 0}';'10^+1';'10^+2';'10^+3';'10^+4';'10^+5';'10^+6';'10^+7';'10^+8';'10^+9'])
legend('Gradient ascent','Accelerated gradient ascent','Newton')
xlabel('Iteration step')
ylabel('Error')
title('Convergence of solutions')
    