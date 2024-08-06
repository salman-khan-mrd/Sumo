function [Totalfeature1]= pssmrequired1(filename)
% for i=1:2059
% 
% %fileinitial='pssMtrx';
% filename=xlsread([num2str(i),'.xls']);
% % matrix=importdata(filename);
% save Matrix matrix;
% load Matrix;
E=[];
mat=[];
matrix=[];
Totalfeature=[];
Totalfeature1=[];
le=size(filename,1);

for i=1:le
avg=mean(filename(i,1:20));
st=std(filename(i,1:20));
for j=1:20
    matrix(i,j)=(filename(i,j)-avg)/st;
end
end
filename1=find(isnan(matrix));
matrix(filename1)=0;

%aq=cell2mat(matrix);
%for j=1:size(matrix.data,1)
  for k=1:20
     a=sum(matrix(:,k));
     E(k)=a/size(matrix,1);
     
  end
%   for n=1:20
%       temp=0;
%       for m=1:size(matrix.data,1)-lambda
%          temp=temp+(matrix.data(m,n)-matrix.data(m+lambda,n))^2;
%       end;
%       %/size(matrix.data,1);
%   end;
%   E(n+20)=temp/size(matrix.data,1)-lambda;
  

for m=1:49
  for n=1:20
    temp=0;
    b=size(matrix,1)-m;
    for j=1:b
        eval=matrix(j,n)-matrix(j+m,n);
        temp=temp+(eval)^2;
    end
 mat(n)=temp/b;
 
  end
Totalfeature=[Totalfeature mat];
end
Totalfeature1=[E Totalfeature]
end

