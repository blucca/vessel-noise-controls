import os
"""Export the frozen pilot candidate; predeclared per-window-only missing-value handling."""
import json
import joblib
import numpy as np
import pandas as pd
from pilot import D,windows,simulate,train_refs,held_refs,features,ExtraTreesRegressor,bands
raw,days,_=windows('acoustics/02_Durban.parquet')
x,y,_,_=simulate(raw,days,train_refs,16000,192)
model=ExtraTreesRegressor(n_estimators=80,max_depth=16,min_samples_leaf=6,n_jobs=1,random_state=192)
model.fit(features(x),y);prior=float(y.mean())
# Verify the exported fit reproduces the already frozen candidate on heldout rows.
held,hd,_=windows('acoustics/10_Sea_Point.parquet');check,_,_,_=simulate(held,hd,held_refs,4000,194)
old=np.load(D/'predictions-10_Sea_Point.npz')['candidate']
assert np.allclose(model.predict(features(check[:16])),old[:16],atol=1e-12)
joblib.dump({'model':model,'prior':prior,'bands':bands,'seed':192},D/'frozen-model.joblib',compress=3)
# Fixed before reading Test: observed within-window band medians; interpolate wholly absent
# bands along log-frequency index. A wholly absent window uses the learned training prior.
test=pd.read_csv(D/'Test.csv'); sample=pd.read_csv(D/'sample_submission.csv')
cols=[f'm{m:02d}_{band}' for m in range(1,16) for band in bands]
assert set(test.columns)==set(['ID']+cols) and test.ID.is_unique and set(test.ID)==set(sample.ID)
db=test[cols].to_numpy(dtype='float32').reshape(-1,15,25)
missing=~np.isfinite(db); empty=missing.all(axis=(1,2));cells_missing=int(missing.sum())
for i in range(len(db)):
 if empty[i]:db[i]=0;continue
 levels=np.array([np.median(db[i,np.isfinite(db[i,:,b]),b]) if np.isfinite(db[i,:,b]).any() else np.nan for b in range(25)])
 known=np.flatnonzero(np.isfinite(levels));filled=np.interp(np.arange(25),known,levels[known])
 for b in range(25):db[i,missing[i,:,b],b]=filled[b]
pred=np.clip(model.predict(features(db)),0,1);pred[empty]=prior
output=pd.DataFrame({'ID':test.ID,'share_added_vessel':pred,'score_added_vessel':pred})
assert list(output.columns)==['ID','share_added_vessel','score_added_vessel'] and np.isfinite(pred).all()
output.to_csv(D/'fixed-submission.csv',index=False)
meta={'source':'frozen pilot candidate; no Test fitting or calibration','rows':len(output),'missing_cells':cells_missing,'wholly_missing_windows':int(empty.sum()),'model_bytes':(D/'frozen-model.joblib').stat().st_size,'prediction_min_median_max':np.quantile(pred,[0,.5,1]).tolist(),'score_added_vessel':'monotone uncalibrated evidence score, identical to the fraction estimate','missing_policy':'per-window band medians and log-frequency-index interpolation, all-missing -> training prior; fixed before reading Test','candidate_reproduction_check':'16 heldout predictions match frozen pilot to atol 1e-12'}
(D/'entry-export.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
