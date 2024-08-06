clc;   
file=fileread('blastresult4a.txt');    % file reading
   
   homologArray=[];   %array for storing all homologs having greater or equal to 60 % identity
   positives=[];
   identity=[];
   homolog=[];
   homologs=[];
   homologArray=[];
   i=1;
   querystart=strfind(file,'Query=');     %first Query= index
  
  % for i=1:size(querystart,2)          %loop for all queries in file
   sepquery=file(querystart(1):querystart(2));    %from first query1 to Query2 content is saved in sepquery
   query=sepquery(8:strfind(sepquery,'(')-1)
   homologArray=strvcat(homologArray,query);
   
   homologs=strfind(sepquery,'>');    % all homologs starting > is stored in homologs
    identity=strfind(sepquery,'Identities ');
     positives=strfind(sepquery,'Positives ');
     length=strfind(sepquery,'Length ');
    for j=1:size(homologs,2)
     homolog=sepquery(homologs(j)+1:length(j)-2)
     identitystr=sepquery(identity(j):positives(j))
     percentidentity=identitystr(strfind(identitystr,'(')+1:strfind(identitystr,')')-2)
     perIDentity=str2num(percentidentity)
 % query=sepquery(homologs(1):strfind(sepquery,' '))%done,
      if(perIDentity>=60)
       homologArray=[homologArray; homolog];
      end
      i=i+1;
    end
    %save homologArray.mat;
 % end;
  


a



