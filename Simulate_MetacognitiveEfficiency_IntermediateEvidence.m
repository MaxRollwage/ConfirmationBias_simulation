clear all
close all
%% define the modelling parameters for all models
samples=10000; %amount of samples per evidence strength and per binary choice option

%% metaD settings
rho_max=.8; 
metacognition_all=[.7 .8 .9 1 1 1 1 1]; % set different values of sigma_conf
rho_all=[rho_max rho_max rho_max rho_max .65 .5 .3 .1]; % set different values of rho


theta2_steps=[.3:.1:1.2]; % define possible reliabilities of pre-decision evidence 
theta1_steps=[.3:.1:1.2]; % define possible reliabilities of post-decision evidence 

%% define specific model parameter for this simulation

model=2;
c_bias=10;
theta1_it=6;
theta2_it=3;

for sub=1:100 % simulate multiple agents for each setting
    for m=1:length(metacognition_all) % loop over all metacognitive abilities
metaD=metacognition_all(m);
rho=rho_all(m);

%% define evidence strength 
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

%% define worldstate
worldstate_left=repmat(-1,samples,1);
worldstate_right=repmat(1,samples,1);
worldstate=[worldstate_left; worldstate_right];
%%sample pre-decision evidence

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


 %confidence-weighted  confirmation bias

  conf_amplifi_confirm=1+((amplifi_confirm-1)*((confidence_initial'-.5)*2));
  conf_amplifi_disconfirm=1+((amplifi_disconfirm-1)*((confidence_initial'-.5)*2));

        
    index_confirm=find((choice_initial==1 & logDir_post>0) | (choice_initial==-1 & logDir_post<0));
    index_disconfirm=find((choice_initial==-1 & logDir_post>0) | (choice_initial==1 & logDir_post<0));

    log_Dir_final(index_confirm)= logDir_pre(index_confirm)+conf_amplifi_confirm(index_confirm).*logDir_post(index_confirm);
    log_Dir_final(index_disconfirm)= logDir_pre(index_disconfirm)+conf_amplifi_disconfirm(index_disconfirm).*logDir_post(index_disconfirm);
    

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
Mean_accuracy_final_no(m,sub)=mean(accuracy_final_no);

%% calculate final performance biased agents
accuracy_final=choice_final==worldstate;
Mean_accuracy_final(m,sub)=mean(accuracy_final);

    end
end
save('results_mid_evidence')

%% calculate group statistics over all simulated agents 
mean_bias=mean(Mean_accuracy_final,2)*100
mean_no_bias=mean(mean(Mean_accuracy_final_no))*100
std_bias=std(Mean_accuracy_final,[],2)/sqrt(sub)*100
std_no_bias=mean(std(Mean_accuracy_final_no,[],2)/sqrt(sub))*100

%% Test metacognitive agents against unbiased agents for signifcant differences

[h,p]=ttest2(Mean_accuracy_final(1,:), Mean_accuracy_final_no(1,:))
[h,p]=ttest2(Mean_accuracy_final(2,:), Mean_accuracy_final_no(2,:))
[h,p]=ttest2(Mean_accuracy_final(3,:), Mean_accuracy_final_no(3,:))
[h,p]=ttest2(Mean_accuracy_final(4,:), Mean_accuracy_final_no(4,:))
[h,p]=ttest2(Mean_accuracy_final(5,:), Mean_accuracy_final_no(5,:))
[h,p]=ttest2(Mean_accuracy_final(6,:), Mean_accuracy_final_no(6,:))
[h,p]=ttest2(Mean_accuracy_final(7,:), Mean_accuracy_final_no(7,:))
[h,p]=ttest2(Mean_accuracy_final(8,:), Mean_accuracy_final_no(8,:))


%% Figure 2B

figure(1)
hold on
biased=bar([1:length(mean_bias)]+2, [mean_bias],.7)
unbiased=bar([1], [mean_no_bias],.7)
errorbar([1:length(mean_bias)]+2, [mean_bias], [std_bias],'.','Color', [0 0 0], 'MarkerSize',2,'MarkerFaceColor',   [0 0 0],'LineWidth',1.5)
errorbar([1], [mean_no_bias], [std_no_bias],'.','Color', [0 0 0], 'MarkerSize',2,'MarkerFaceColor',   [0 0 0],'LineWidth',1.5)
ylim([81, 85])
ylabel('Performance (%correct)')
legend([unbiased, biased], {'unbiased agent', 'metacognitive agent'},'Location','northwest','box','off')
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[3:10], 'XTickLabel',{'0.4','0.6','0.8','1','1.1','1.2','1.3','1.4'})
set(findall(gca, 'Type', 'Line'),'LineWidth',1.5)
