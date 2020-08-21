# ConfirmatioBias_simulation
This repository contains the code for the simulations presented in the following paper: Rollwage &amp; Fleming (2020) Confirmation bias is adaptive when coupled with self-awareness. Philosophical Transactions B

The script Simulate_ConfirmationBias_ConfidenceWeighting.m reproduces the simulations presented in Figure 1, comparing performance of an unbiased agents compared to an agent with confirmation bias or a confidence-weighted confirmation bias.

The script Simulate_MetacognitiveEfficiency.m reproduces the simulations presented in Figure 2A&C, showing that the performance of agents with confidence-weighted confirmation bias depends on their metaocgnitive efficiecny


The script Simulate_MetacognitiveEfficiency_IntermediateEvidence.m reproduces the simulations presented in Figure 2B, which enables the quantification of the effect (influence of metacognition on performance) for a setting with intermediate evidence strength which is comparable to evidence strengths commonly used in experiments using human participants. 

All scripts use the ehlper function prepare_metaD.m for calculating metacognitive abilities. Moreover, for fitting meta-d the MATLAB code from Maniscalco & Lau’s   (http://www.columbia.edu/~bsm2105/type2sdt/) is required. 
