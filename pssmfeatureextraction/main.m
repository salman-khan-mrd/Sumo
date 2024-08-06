clc;

PSSM_anticancer=[];
for i=1:321
i
%fileinitial='pssMtrx';
filename=xlsread([num2str(i),'.xls']);
PSSM_anticancer=[PSSM_anticancer; pssmrequired1(filename)];
end
save PSSM_anticancer PSSM_anticancer;
% load PSSM_Feature_2059;
% for i=1:2059
%     for j=1:40
%         strcmp(PSSM_feature(i,j)=='NaN')
%             PSSM_feature(i,j)=0;
%         
%     end
% end
% a