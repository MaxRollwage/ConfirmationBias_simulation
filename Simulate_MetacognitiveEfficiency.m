clear all
close all
%% define the modelling parameters for all models
samples=10000; %amount of samples per evidence strength and per binary choice option

%% metaD settings

metacognition_all=[.7 .8 .9 1 1 1 1 1]; % set different values of sigma_conf
rho_all=[rho_max rho_max rho_max rho_max .65 .5 .3 .1]; % set different values of rho


theta2_steps=[.3:.1:1.2]; % define possible reliabilities of pre-decision evidence 
theta1_steps=[.3:.1:1.2]; % define possible reliabilities of post-decision evidence 



for m=1:length(metacognition_all) % loop over all metacognitive abilities
metaD=metacognition_all(m);
rho=rho_all(m);
for model=1:2 % loop over 1= confirmation bias; 2=metacognitive agent
    c_bias_it=0

for c_bias=5:.5:10 % loop over confirmation bias strength, from 5(no bias) to 10(full confirmation bias)
c_bias_it=c_bias_it+1
for theta1_it=1:10 %loop over all pre-decision evidence strengths
    for theta2_it=1:10 %loop over all post-decision evidence strengths
theta1=theta1_steps(theta1_it);
theta2=theta2_steps(theta2_it);

mu = [1;  1];
mu_post=1;

sigma_act=mu(1)/theta1;
sigma_conf=mu(2)/(theta1*metaD);
sigma = [sigma_act^2 rho*sigma_conf*sigma_act; rho*sigma_conf*sigma_act sigma_conf^2];
sigma_post=mu_post./theta2;

%% define worldstate
worldstate_left=repmat(-1,samples,1);
worldstate_right=repmat(1,samples,1);
worldstate=[worldstate_left; worldstate_right];
%%  sample pre-decision evidence

[X_left]=mvnrnd(mu.*-1,sigma, samples);
Xpre_left=X_left(:,1);
Xconf_left=X_left(:,2);

[X_right]=mvnrnd(mu,sigma, samples);
Xpre_right=X_right(:,1);
Xconf_right=X_right(:,2);


Xpre=[Xpre_left; Xpre_right];
Xconf=[Xconf_left; Xconf_right];


%% convert pre-decision evidence into initial decision and confidence
logDir_pre=(2*mu(1)*Xpre)/(sigma_act^2);
choice_initial=ones(length(Xpre),1); 
choice_initial(logDir_pre<=0)=-1;

index_left=find(choice_initial==-1);
logConf_pre=(2*mu(2)*Xconf)./(sigma_conf^2);
logConf_pre(index_left)=-logConf_pre(index_left);

%% calculate confidence based on the initial decision, Xconf and their covariance
for k=1:length(Xconf)
confidence_initial(k) = computeMetaConf(Xconf(k), choice_initial(k), sigma_act, sigma_conf, rho);
end


%% sample post-decision evidence
Xpost=normrnd(mu_post.*worldstate,sigma_post, length(worldstate), 1);

%% convert post-decision evidence into log-odds

logDir_post=(2*mu_post*Xpost)./(sigma_post^2);


%% define strength of confirmation bias
confirmation_bias=c_bias/10; 
amplifi_confirm=(confirmation_bias)*2;
amplifi_disconfirm=(1-confirmation_bias)*2;


%% combine pre-and post-decision evidence for final judgment
% unbiased Model
log_Dir_final_no= logDir_pre+logDir_post;
choice_final_no=ones(length(log_Dir_final_no),1); 
choice_final_no(log_Dir_final_no<0)=-1;

%biased models
if model==1
  %simple confirmation bias
   
    index_confirm=find((choice_initial==1 & logDir_post>0) | (choice_initial==-1 & logDir_post<0));
    index_disconfirm=find((choice_initial==-1 & logDir_post>0) | (choice_initial==1 & logDir_post<0));

    log_Dir_final(index_confirm)= logDir_pre(index_confirm)+amplifi_confirm*logDir_post(index_confirm);
    log_Dir_final(index_disconfirm)= logDir_pre(index_disconfirm)+amplifi_disconfirm*logDir_post(index_disconfirm);

    
elseif model==2
      %confidence-weighted  confirmation bias

    conf_amplifi_confirm=1+((amplifi_confirm-1)*((confidence_initial'-.5)*2));
    conf_amplifi_disconfirm=1+((amplifi_disconfirm-1)*((confidence_initial'-.5)*2));

        
    index_confirm=find((choice_initial==1 & logDir_post>0) | (choice_initial==-1 & logDir_post<0));
    index_disconfirm=find((choice_initial==-1 & logDir_post>0) | (choice_initial==1 & logDir_post<0));

    log_Dir_final(index_confirm)= logDir_pre(index_confirm)+conf_amplifi_confirm(index_confirm).*logDir_post(index_confirm);
    log_Dir_final(index_disconfirm)= logDir_pre(index_disconfirm)+conf_amplifi_disconfirm(index_disconfirm).*logDir_post(index_disconfirm);
    

end


%% derive final choice and confidence
choice_final=ones(length(log_Dir_final),1); 
choice_final(log_Dir_final<0)=-1;

index_left=find(choice_final==-1);
log_Dir_final(index_left)=-log_Dir_final(index_left);
confidence_final=exp(log_Dir_final)./(1+exp(log_Dir_final)); 

%% calculate initial performance 
accuracy_initial=choice_initial==worldstate;
Mean_accuracy_inital(theta1_it, theta2_it ,model)=mean(accuracy_initial);

%% calculate final performance unbiased agent
accuracy_final_no=choice_final_no==worldstate;
Mean_accuracy_final_no(theta1_it, theta2_it)=mean(accuracy_final_no);

%% calculate final performance biased agents
accuracy_final=choice_final==worldstate;
Mean_accuracy_final(theta1_it, theta2_it, model)=mean(accuracy_final);

if model==2
    % calculate the metacognitive ability for each model setting and each
    % evidence strength
[fit]=prepare_metaD(worldstate, choice_initial, confidence_initial');
meta_d(theta1_it, theta2_it)=fit.meta_da;
D_prime(theta1_it, theta2_it)=fit.da;
end
    
    
end
end

%% average performance for unbiased agent over all evidence strength

Mean_accuracy_baseline(model, c_bias_it, m)=mean(mean(Mean_accuracy_final(:, :, model)-Mean_accuracy_final_no));
if model==2
%% calculate metacognitive abilities averaged over all evidence strengths

meta_d_average(c_bias_it, m)=mean(mean(meta_d)); 
D_prime_average(c_bias_it, m)=mean(mean(D_prime)); 
end
end
end
end
%% calculate metacognitive efficiency for each agent
meta_efficiency=round(mean(meta_d_average./D_prime_average).*100)./100;

%% Figure 2A
figure(1)
h=heatmap(flipud(squeeze(Mean_accuracy_baseline(2,:, :))*100), 'ColorbarVisible', 'on')
ylabel('Confirmation bias strength')
 h = colorbar;
colormap parula
caxis([-1 1]);
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[1,3, 5, 7, 9, 11], 'YTickLabel',fliplr({'0','0.2','0.4','0.6','0.8','1'}), 'XTick',[1:8], 'XTickLabel',{num2str(meta_efficiency_use(1)),num2str(meta_efficiency_use(2)), num2str(meta_efficiency_use(3)) , num2str(meta_efficiency_use(4)), num2str(meta_efficiency_use(5)), num2str(meta_efficiency_use(6)), num2str(meta_efficiency_use(7)), num2str(meta_efficiency_use(8))})


%% calculate the difference in perforance between a simple confirmation biasand a confidence-weighted confirmation bias
diff_biases=squeeze(Mean_accuracy_baseline(2,:, :)-Mean_accuracy_baseline(1,:, :))*100

%% Figure 2C

figure(2)
h=heatmap(flipud(diff_biases), 'ColorbarVisible', 'on')
ylabel('Confirmation bias strength')
 h = colorbar;
colormap parula
caxis([-1 10]);
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[1,3, 5, 7, 9, 11], 'YTickLabel',fliplr({'0','0.2','0.4','0.6','0.8','1'}), 'XTick',[1:8], 'XTickLabel',{num2str(meta_efficiency_use(1)),num2str(meta_efficiency_use(2)), num2str(meta_efficiency_use(3)) , num2str(meta_efficiency_use(4)), num2str(meta_efficiency_use(5)), num2str(meta_efficiency_use(6)), num2str(meta_efficiency_use(7)), num2str(meta_efficiency_use(8))})
