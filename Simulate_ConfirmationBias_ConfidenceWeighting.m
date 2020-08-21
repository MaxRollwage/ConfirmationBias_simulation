ticlear all
close all
%% define the modelling parameters for all models
samples=10000; %amount of samples per evidence strength and per binary choice option

%% metaD settings
rho=.999999; % due to numerical issues this value has to be .9999 instead of 1
metaD=1;

theta2_steps=[.3:.1:1.2]; % define possible reliabilities of pre-decision evidence 
theta1_steps=[.3:.1:1.2]; % define possible reliabilities of post-decision evidence 

for model=1:2 % loop over 1= confirmation bias; 2=metacognitive agent
    c_bias_it=0

for c_bias=5:10 % loop over confirmation bias strength, from .5(no bias) to 1(full confirmation bias)
c_bias_it=c_bias_it+1
for theta1_it=1:10 %loop over all pre-decision evidence strengths
    for theta2_it=1:10 %loop over all post-decision evidence strengths

%% define pre and post-decision evidence strength
theta1=theta1_steps(theta1_it);
theta2=theta2_steps(theta2_it);
mu = [1;  1];
mu_post=1;

sigma_post=mu_post./theta2;
sigma_act=mu(1)/theta1;
sigma_conf=mu(2)/(theta1*metaD);
sigma = [sigma_act^2 rho*sigma_conf*sigma_act; rho*sigma_conf*sigma_act sigma_conf^2];

%% define worldstate
worldstate=randi([0 1],samples,1);
worldstate(worldstate==0)=-1;

%% sample pre-decision evidence
[X]=mvnrnd(mu,sigma, samples);

Xpre=X(:,1);
Xconf=X(:,2);
Xpre=Xpre.*worldstate;
Xconf=Xconf.*worldstate;

%% convert pre-decision evidence into initial decision and confidence
logDir_pre=(2*mu(1)*Xpre)/(sigma_act^2);
choice_initial=ones(length(Xpre),1); 
choice_initial(logDir_pre<0)=-1;

index_left=find(choice_initial==-1);
logConf_pre=(2*mu(2)*Xconf)./(sigma_conf^2);
logConf_pre(index_left)=-logConf_pre(index_left);
confidence_initial=exp(logConf_pre)./(1+exp(logConf_pre)); 


%% sample post-decision evidence
Xpost=normrnd(mu_post.*worldstate,sigma_post, samples, 1);

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
  
    index_confirm=find((choice_initial==1 & logDir_post>0) | (choice_initial==-1 & logDir_post<0));
    index_disconfirm=find((choice_initial==-1 & logDir_post>0) | (choice_initial==1 & logDir_post<0));

    log_Dir_final(index_confirm)= logDir_pre(index_confirm)+amplifi_confirm*logDir_post(index_confirm);
    log_Dir_final(index_disconfirm)= logDir_pre(index_disconfirm)+amplifi_disconfirm*logDir_post(index_disconfirm);

elseif model==2
  conf_amplifi_confirm=1+((amplifi_confirm-1)*((confidence_initial-.5)*2));
  conf_amplifi_disconfirm=1+((amplifi_disconfirm-1)*((confidence_initial-.5)*2));

        
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


if c_bias_it==5
    if model==1
    Example_confirmationBias=Mean_accuracy_final(:, :, 1)-Mean_accuracy_final_no(:,:);
    else
    Example_confidence_weighted=Mean_accuracy_final(:, :, 2)-Mean_accuracy_final_no(:,:);
    end    
    
end


    end
end

%% average performance for unbiased agent over all evidence strength
Mean_accuracy_baseline(model, c_bias_it)=mean(mean(Mean_accuracy_final(:, :, model)-Mean_accuracy_final_no));

end
end

mean_no_bias=repmat(mean(mean(Mean_accuracy_final_no)), 1, 6)



%% Figure 1A
figure(2)
h=heatmap(flipud(Mean_accuracy_final_no)*100);
title('Unbiased')
colormap copper
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[1:10], 'YTickLabel',fliplr({num2str(theta1_steps(1)),num2str(theta1_steps(2)),num2str(theta1_steps(3)),num2str(theta1_steps(4)),num2str(theta1_steps(5)),num2str(theta1_steps(6)),num2str(theta1_steps(7)),num2str(theta1_steps(8)),num2str(theta1_steps(9)),num2str(theta1_steps(10))}), 'XTick',[1:10], 'XTickLabel',{num2str(theta1_steps(1)),num2str(theta1_steps(2)),num2str(theta1_steps(3)),num2str(theta1_steps(4)),num2str(theta1_steps(5)),num2str(theta1_steps(6)),num2str(theta1_steps(7)),num2str(theta1_steps(8)),num2str(theta1_steps(9)),num2str(theta1_steps(10))})
h1 = colorbar;
ylabel(h1, 'Performance (%correct)','FontSize',14)


%% Figure 1B
figure(3)
h=heatmap(flipud(Example_confirmationBias)*100);
title('Confirmation bias')
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[1:10], 'YTickLabel',fliplr({num2str(theta1_steps(1)),num2str(theta1_steps(2)),num2str(theta1_steps(3)),num2str(theta1_steps(4)),num2str(theta1_steps(5)),num2str(theta1_steps(6)),num2str(theta1_steps(7)),num2str(theta1_steps(8)),num2str(theta1_steps(9)),num2str(theta1_steps(10))}), 'XTick',[1:10], 'XTickLabel',{num2str(theta1_steps(1)),num2str(theta1_steps(2)),num2str(theta1_steps(3)),num2str(theta1_steps(4)),num2str(theta1_steps(5)),num2str(theta1_steps(6)),num2str(theta1_steps(7)),num2str(theta1_steps(8)),num2str(theta1_steps(9)),num2str(theta1_steps(10))})
h = colorbar;
colormap bone
caxis([-7 0]);
ylabel(h, 'Detrimental performance (%correct)','FontSize', 14)

%% Figure 1C
figure(1)
hold on
no_bias=plot([1:length(Mean_accuracy_baseline(1,:, 1))], mean_no_bias,'LineWidth',3)
confirmation_bias=plot([1:length(Mean_accuracy_baseline(1,:, 1))], Mean_accuracy_baseline(1,:, 1)+mean_no_bias,'LineWidth',3)
confidenec_weighted=plot([1:length(Mean_accuracy_baseline(2,:, 1))], Mean_accuracy_baseline(2,:, 1)+mean_no_bias,'LineWidth',3)
legend([no_bias, confidenec_weighted,confirmation_bias], {'unbiased','confidence weighted','confirmation bias', },'Location', 'southwest')
xlabel('confirmation bias strength')
ylabel('Performance (%correct)')
ylim([.75 .86])
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off','YTick',[.75,.775,.8,.825,.85], 'YTickLabel',{'75%','77.5%','80%','82.5%','85%'}, 'XTick',[1:6], 'XTickLabel',{'0','.2','.4','.6','.8','1'})


%% Figure 1D
figure(4)
h=heatmap(flipud(Example_confidence_weighted)*100);
title('Confidence weighted confirmation bias')
set(gca, 'FontSize', 14,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[1:10], 'YTickLabel',fliplr({num2str(theta1_steps(1)),num2str(theta1_steps(2)),num2str(theta1_steps(3)),num2str(theta1_steps(4)),num2str(theta1_steps(5)),num2str(theta1_steps(6)),num2str(theta1_steps(7)),num2str(theta1_steps(8)),num2str(theta1_steps(9)),num2str(theta1_steps(10))}), 'XTick',[1:10], 'XTickLabel',{num2str(theta1_steps(1)),num2str(theta1_steps(2)),num2str(theta1_steps(3)),num2str(theta1_steps(4)),num2str(theta1_steps(5)),num2str(theta1_steps(6)),num2str(theta1_steps(7)),num2str(theta1_steps(8)),num2str(theta1_steps(9)),num2str(theta1_steps(10))})
 h = colorbar;
 colormap bone
caxis([-7 0]);
ylabel(h, 'Detrimental performance (%correct)','FontSize',14)

