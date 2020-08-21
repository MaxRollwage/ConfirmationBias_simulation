function [fit]=prepare_metaD(worldstate, choice_initial, confidence_initial)

%% calculate meta-d
median_confidence=median(confidence_initial);


low_left_left=length(find(worldstate==-1 & choice_initial==-1 & confidence_initial<median_confidence));
high_left_left=length(find(worldstate==-1 & choice_initial==-1 & confidence_initial>=median_confidence));
low_left_right=length(find(worldstate==-1 & choice_initial==1 & confidence_initial<median_confidence));
high_left_right=length(find(worldstate==-1 & choice_initial==1 & confidence_initial>=median_confidence));

low_right_right=length(find(worldstate==1 & choice_initial==1 & confidence_initial<median_confidence));
high_right_right=length(find(worldstate==1 & choice_initial==1 & confidence_initial>=median_confidence));
low_right_left=length(find(worldstate==1 & choice_initial==-1 & confidence_initial<median_confidence));
high_right_left=length(find(worldstate==1 & choice_initial==-1 & confidence_initial>=median_confidence));

nR_S1=[high_left_left low_left_left low_left_right high_left_right];
nR_S2=[high_right_left low_right_left low_right_right high_right_right];

 adj_f = 1/length(nR_S1);
 nR_S1_adj = nR_S1 + adj_f;
 nR_S2_adj = nR_S2 + adj_f;

fit = fit_meta_d_MLE(nR_S1_adj, nR_S2_adj);
