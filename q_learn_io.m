function [u_q,z_bar,info] = q_learn_io(N,u,y,Q,R,df,u_bar,y_bar)
 %%
m = size(u,1); %dimensi input 
p = size(y,1); %dimensi output
mm = size(u,2);
% for i =1:N
%     z_bar(:,i) = [u(i);y(i)];
% end
for i =1:N
    Z_bar(i,:) = [u(:,i)' u_bar y_bar];
    % z_bar(i,:) = reshape(Z_bar(i,:),[i,m+p])';
end
% z_bar = z_bar(:,1);
z_bar = Z_bar;
% z_bar = reshape(Z_bar(:,1),[N,m+p])';
%%
tic 
% Q = 1*eye(3);
% R = 0.1;
iter = N;
% u_bar = u_bar(1,1:N-1);
% y_bar = y_bar(1,1:N);
% P = eye(202,202);
P = rand(m*N+p*N+m,m*N+p*N+m);
p0 = rand(m,m);
pu = rand(m,m*(N));
py = rand(m,p*N);
p22 = rand(m*(N-1),m*(N-1));
p23 = rand(m*(N-1),p*N); 
p32 = rand(p*N,m*(N-1));
p33 = rand(p*N,p*N);
% P = [p0 pu py;pu' p22 p23;py' p32 p33];
u = zeros(m,N);
%%
% Q = 1;R = 10;
    for i = 1:iter
        if i~=1
            % Policy Evaluation 
            % P(:,:,i) = y(:,i)*Q*y(:,i)'+u(:,i)*R*u(:,1)'+df*Z_bar(i,:)*Z_bar(i,:)'*P(:,:,i-1);
            for j = 1:mm
                P(:,:,i) = y(:,j)'*y(:,j)*Q+u(:,j)'*u(:,j)*R+z_bar(i,:)*z_bar(i,:)'*P(:,:,i-1);
            % P(:,:,i) = y_bar*y_bar'*Q+u_bar*u_bar'*R+Z_bar'*Z_bar*P(:,:,i-1);
            % Policy Improvement
            p0(:,:,i) = P(1:m,1:m,i);
            pu(:,:,i) = P(1:m,m+1:m+(m*N),i);
            py(:,:,i) = P(1:m,(m*N)+1:(m+p)*N);
            u(:,i+1) = -df*inv(R+df*p0(:,:,i))*(pu(:,:,i)*u_bar'+py(:,:,i)*y_bar');
%             u(i+1) = -df*inv(R+p0(i))*(pu(i,:)*u_bar+py(i,:)*y_bar);
            end
        end
    end
    %%
for i = 1:iter
    P_norm(i) = norm(P(:,:,i));
end
time_elapsed = toc;
info.time = time_elapsed;
info.kernelP = P;
info.normP = P_norm;
u_q = u;