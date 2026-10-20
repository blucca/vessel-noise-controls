# Frozen pilot decision

Before reading any model results, add one reviewer-requested matched comparator: an identical ExtraTrees model using the four gain-invariant temporal-energy features. Candidate uses relative band shape, band spreads and those temporal features. Compare against the best of this baseline, absolute-energy baseline and the training-prior constant.

Per heldout station require >=10% RMSE gain, paired background-day/reference-hull bootstrap lower gain bound >0, and on known zero controlled-addition synthetic rows require mean predicted fraction <0.10 and p90 <0.30. These support a research continuation decision, not natural vessel-presence validation. +12dB invariance is a construction check with no independent evidence bonus.

No hyperparameter or simulator sweep; one preselected model/simulator; reference/hull/day and South African station holdouts; Test untouched. The pre-comparator run, if already started, is superseded without using its outputs for tuning.
