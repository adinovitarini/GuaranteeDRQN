function [iter,perfIndex] = PerfAnalysis(u_q,y_q,Yref,Q,R)
%%
perfIndex = sum(y_q'*y_q*Q+u_q'*u_q*R);
iter = 0;
N = size(y_q,2);
% for i = 1:N 
    % if (y_q(:,i)-Yref>=1e-20)
        iter = iter+1;
        % break
    % end
% end