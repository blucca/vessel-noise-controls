import os
"""One frozen simulator/estimator pilot. Released training data only; no test input."""
import io,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
D=Path(os.environ.get('VESSEL_DATA_DIR','temp/vessel-noise')); RNG=np.random.default_rng(192)
refs=np.load(D/'references.npz')['bands'].mean(axis=1)
ref_meta=json.loads((D/'reference-audit.json').read_text())
train_refs=np.array([i for i,r in enumerate(ref_meta) if r['split']=='train'])
held_refs=np.array([i for i,r in enumerate(ref_meta) if r['split']=='heldout'])
background_refs=np.array([i for i,r in enumerate(ref_meta) if r['split']=='background_control'])
assert {ref_meta[i]['hull_id'] for i in train_refs}.isdisjoint({ref_meta[i]['hull_id'] for i in held_refs})
bands=['TOL_31.6', 'TOL_39.8', 'TOL_50.1', 'TOL_63.1', 'TOL_79.4', 'TOL_100', 'TOL_125.9', 'TOL_158.5', 'TOL_199.5', 'TOL_251.2', 'TOL_316.2', 'TOL_398.1', 'TOL_501.2', 'TOL_631', 'TOL_794.3', 'TOL_1000', 'TOL_1258.9', 'TOL_1584.9', 'TOL_1995.3', 'TOL_2511.9', 'TOL_3162.3', 'TOL_3981.1', 'TOL_5011.9', 'TOL_6309.6', 'TOL_7943.3']

def windows(member):
 with zipfile.ZipFile(D/'South_African_Real_Acoustics.zip') as z:
  table=pq.read_table(io.BytesIO(z.read(member)),columns=['recorder','t','dur']+bands)
 df=table.to_pandas().sort_values(['recorder','t']).reset_index(drop=True)
 x=df[bands].to_numpy(dtype=np.float32)
 valid=np.isfinite(x).all(axis=1)&(df.dur.to_numpy()>=55)
 gaps=df.t.diff().dt.total_seconds().to_numpy()
 cut=np.r_[True,(df.recorder.to_numpy()[1:]!=df.recorder.to_numpy()[:-1])|~valid[1:]|~valid[:-1]|(gaps[1:]<59)|(gaps[1:]>61)]
 bounds=np.r_[np.flatnonzero(cut),len(df)]; blocks=[]; starts=[]; days=[]
 for a,b in zip(bounds[:-1],bounds[1:]):
  if not valid[a]:continue
  n=(b-a)//15
  if n:
   blocks.append(x[a:a+n*15].reshape(n,15,25)); starts.extend(str(t) for t in df.t.iloc[a:a+n*15:15]);days.extend(str(t.date()) for t in df.t.iloc[a:a+n*15:15])
 out=np.concatenate(blocks)
 audit={'member':member,'source_rows':len(df),'finite_duration_rows':int(valid.sum()),'nonoverlap_windows':len(out),'days':len(set(days)),'first_window':starts[0],'last_window':starts[-1]}
 return out,np.array(days),audit

def energy(db):
 anchor=db.max(axis=(1,2),keepdims=True)
 return 10.0**((db-anchor)/10),anchor

def features(db,kind='candidate'):
 e,anchor=energy(db); bytime=e.sum(axis=2); total=e.mean(axis=(1,2)); logtime=10*np.log10(bytime+1e-30)
 qt=np.quantile(logtime,[.1,.5,.9],axis=1).T
 temporal=np.column_stack((qt[:,2]-qt[:,0],logtime.max(axis=1)-qt[:,1],np.abs(np.diff(logtime,axis=1)).mean(axis=1),logtime.std(axis=1)))
 if kind=='energy':return np.column_stack((anchor[:,0,0]+10*np.log10(total),qt+anchor[:,0,0,None],temporal)).astype('float32')
 shape=10*np.log10(e.mean(axis=1)+1e-30);shape-=shape.mean(axis=1,keepdims=True)
 spread=np.quantile(db,.9,axis=1)-np.quantile(db,.1,axis=1)
 return np.column_stack((shape,spread,temporal)).astype('float32')

def simulate(raw,days,ref_ids,n,seed,stress=False):
 rng=np.random.default_rng(seed); wi=rng.integers(0,len(raw),size=n);ri=rng.choice(ref_ids,size=n)
 bg,anchor=energy(raw[wi]); template=refs[ri].copy()
 # A frozen, declared surrogate envelope; two-minute excerpts provide spectral references.
 slope=rng.uniform(-2,2,size=(n,1)) if not stress else rng.uniform(-5,5,size=(n,1))
 template*=10**(slope*np.linspace(-1,1,25)[None,:]/10)
 center=rng.uniform(0,14,size=n);width=rng.uniform(1.5,6,size=n) if not stress else rng.uniform(.75,9,size=n)
 envelope=np.exp(-.5*((np.arange(15)[None,:]-center[:,None])/width[:,None])**2)
 v=envelope[:,:,None]*template[:,None,:];v/=v.mean(axis=(1,2),keepdims=True)
 mode=rng.random(n); fraction=rng.uniform(.05,.7,size=n);fraction[mode<.30]=0
 v*=bg.mean(axis=(1,2),keepdims=True)*(fraction/(1-fraction))[:,None,None]
 mixed=bg+v
 # Pure amplitude modulation: a known no-passage-addition control, target remains zero.
 nuisance=(mode>=.15)&(mode<.30)
 mixed[nuisance]*=10**(rng.uniform(0,10,size=(nuisance.sum(),1,1))*envelope[nuisance,:,None]/10)
 db=(anchor+10*np.log10(np.maximum(mixed,1e-30))).astype('float32')
 return db,fraction,days[wi],np.array([ref_meta[i]['hull_id'] for i in ri])

def paired_bootstrap(y,p,b,days,hulls,seed=192):
 _,di=np.unique(days,return_inverse=True);_,hi=np.unique(hulls,return_inverse=True);rng=np.random.default_rng(seed);values=[]
 err=(p-y)**2;berr=(b-y)**2
 for _ in range(240):
  w=rng.poisson(1,di.max()+1)[di]*rng.poisson(1,hi.max()+1)[hi]
  if w.sum():values.append(np.sqrt(np.average(berr,weights=w))-np.sqrt(np.average(err,weights=w)))
 return np.quantile(values,[.025,.975]).tolist()

def summarize(y,p):return {'rmse':float(np.sqrt(np.mean((p-y)**2))),'mae':float(np.mean(abs(p-y))),'pred_mean':float(p.mean())}

def run():
 start=time.monotonic(); train,td,ta=windows('acoustics/02_Durban.parquet'); print('training_windows',ta,flush=True)
 x,y,_,_=simulate(train,td,train_refs,16000,192)
 assert np.allclose(features(x[:3]),features(x[:3]+12),atol=5e-5)
 model=ExtraTreesRegressor(n_estimators=80,max_depth=16,min_samples_leaf=6,n_jobs=1,random_state=192)
 baseline=ExtraTreesRegressor(n_estimators=80,max_depth=16,min_samples_leaf=6,n_jobs=1,random_state=192)
 model.fit(features(x),y);baseline.fit(features(x,'energy'),y)
 temporal_baseline=ExtraTreesRegressor(n_estimators=80,max_depth=16,min_samples_leaf=6,n_jobs=1,random_state=192)
 temporal_baseline.fit(features(x,'energy')[:,-4:],y)
 constant=float(y.mean()); del x
 result={'seed':192,'source_declaration':'https://zindi.world/competitions/french-embassy-underwater-vessel-noise-challenge/discussions/35443','train_windows':ta,'train_simulations':len(y),'training_target_mean':constant,'reference_groups':{'train_hulls':len(train_refs),'heldout_hulls':len(held_refs),'background_control_clips':len(background_refs)},'holdouts':[]}
 for number,member in enumerate(['acoustics/07_St_Francis.parquet','acoustics/10_Sea_Point.parquet']):
  raw,days,audit=windows(member);print('heldout_windows',audit,flush=True)
  generated,target,gd,gh=simulate(raw,days,held_refs,4000,193+number)
  pred=model.predict(features(generated));bp=baseline.predict(features(generated,'energy'));cp=np.full(len(target),constant)
  tp=temporal_baseline.predict(features(generated,'energy')[:,-4:])
  mb=summarize(target,bp);mc=summarize(target,cp);mt=summarize(target,tp);mp=summarize(target,pred)
  best_index=int(np.argmin([mb['rmse'],mc['rmse'],mt['rmse']]))
  bestp=[bp,cp,tp][best_index];best=[mb['rmse'],mc['rmse'],mt['rmse']][best_index]
  record={'station':member,'coverage':audit,'simulator_validation':{'candidate':mp,'energy_baseline':mb,'constant_baseline':mc,'gain_invariant_temporal_baseline':mt,'relative_rmse_improvement':1-mp['rmse']/best,'paired_day_hull_bootstrap_rmse_gain_95ci':paired_bootstrap(target,pred,bestp,gd,gh)}}
  zero=target==0
  record['synthetic_no_added_reference']={'count':int(zero.sum()),'candidate':summarize(target[zero],pred[zero]),'candidate_p90':float(np.quantile(pred[zero],.9)),'best_baseline':summarize(target[zero],bestp[zero])}
  stress,sy,_,_=simulate(raw,days,held_refs,2000,195+number,stress=True)
  record['channel_envelope_stress']=summarize(sy,model.predict(features(stress)))
  # All original heldout windows: no inserted reference, but natural vessel activity is unknown.
  original=model.predict(features(raw)); received=features(raw,'energy')[:,0]
  control={ 'original_pred_mean':float(original.mean()),'original_pred_quantiles':np.quantile(original,[.1,.5,.9]).tolist(),'received_energy_spearman':float(spearmanr(original,received).statistic)}
  gain=model.predict(features(raw+12));control['uniform_12db_gain_max_prediction_change']=float(abs(gain-original).max())
  r=np.random.default_rng(292+number);shuffle=raw.copy()
  for i in range(len(shuffle)):
   for b in range(25):shuffle[i,:,b]=shuffle[i,r.permutation(15),b]
  shuffled=model.predict(features(shuffle));control['independent_band_time_shuffle_mean_delta']=float((shuffled-original).mean());control['independent_band_time_shuffle_mean_abs_delta']=float(abs(shuffled-original).mean())
  reversed_pred=model.predict(features(raw[:,:,::-1]));control['band_reversal_mean_delta']=float((reversed_pred-original).mean())
  # AIS-background source references preserve added energy but carry a different scenario tag.
  bgmix,by,_,_=simulate(raw,days,background_refs,2000,198+number)
  bgpred=model.predict(features(bgmix));mask=by>0
  control['ais_background_reference_additions']={'intended_added_energy_mean':float(by[mask].mean()),'pred_mean':float(bgpred[mask].mean()),'all_reference_energy_rmse':float(np.sqrt(np.mean((bgpred[mask]-by[mask])**2))),'meaning':'weak-label source audit, not a vessel-free ground-truth test'}
  record['real_window_controls']=control
  record['pilot_numeric_gate']=bool(record['simulator_validation']['relative_rmse_improvement']>=.10 and record['simulator_validation']['paired_day_hull_bootstrap_rmse_gain_95ci'][0]>0 and record['synthetic_no_added_reference']['candidate']['pred_mean']<.10 and record['synthetic_no_added_reference']['candidate_p90']<.30)
  result['holdouts'].append(record)
  np.savez_compressed(D/('predictions-'+Path(member).stem+'.npz'),y=target,candidate=pred,energy=bp,constant=cp,temporal=tp,background_day=gd,reference_hull=gh,original=original,received_energy=received,original_day=days,independent_band_time_shuffle=shuffled)
  (D/'pilot-results.json').write_text(json.dumps(result,indent=2));print(json.dumps(record,indent=2),flush=True)
 result['elapsed_seconds']=time.monotonic()-start
 result['both_numeric_gates_pass']=all(x['pilot_numeric_gate'] for x in result['holdouts'])
 (D/'pilot-results.json').write_text(json.dumps(result,indent=2));print('complete_seconds',result['elapsed_seconds'],'gates',result['both_numeric_gates_pass'],flush=True)

if __name__=="__main__":
 run()
