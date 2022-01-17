function [nse,r2, pbias,rmse] = stat( pre, obs )
%calculate NSE
upper=sum((pre-obs).^2);
down=sum((obs-mean(obs).*ones(size(obs,1),1)).^2);
nse=1-upper/down;

%calculate R2
r2=corr(obs,pre)^2;

%calculate PBIAS
upper2=sum(obs-pre);
down2=sum(obs);
pbias=upper2/down2*100;

%calculate RMSE 
rmse=(mean((pre-obs).^2))^(1/2);

end