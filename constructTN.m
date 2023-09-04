function TN = constructTN(sys,N)
A = sys.A;
B = sys.B;
C = sys.C;
m = size(B,2); %jumlah input
p = size(A,1); %jumlah state 
n = size(C,2); %jumlah output
% N = 5;
ka = [1:m:N*m];
kb = [m:m:N*m];
for k = 1:N-1
    temp_r(:,ka(k):kb(k)) = C*A^(k-1)*B;
end
Q2 = {zeros(p,m),temp_r};
%% concat menjadi sebuah matriks kolom
[row1,col1] = size(Q2{1});
[row2,col2] = size(temp_r);
xx = [1:1:(col2+m)/m];
X = tril(toeplitz([xx]));
%%
nn = col2/col1; 
currCol = 1;
endCol = col1;
for i = 1:nn 
    Q2{i+1} = temp_r(:,currCol:endCol);
    currCol = currCol+col1;
    endCol = endCol+col1;
end
%%
X(X==0)=1;
TN = cell2mat(Q2(X));
