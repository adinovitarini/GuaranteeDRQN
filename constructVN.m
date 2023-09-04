%% Compute VN
function VN = constructVN(sys,N)
%%
A = sys.A;
B = sys.B;
C = sys.C;
m = size(B,2); %jumlah input
p = size(A,1); %jumlah state 
n = size(C,2); %jumlah output
ja = [1:n:N*n];
jb = [n:n:N*n];
for j = 1:size(ja,2)
    VN(ja(j):jb(j),:) = C*A^(j-1);
end