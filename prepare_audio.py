import os
"""Deterministic public VTUAD excerpts, following source declaration 35443."""
import csv,hashlib,io,json,struct,time,wave,zlib
from pathlib import Path
import numpy as np
import requests
from scipy.signal import periodogram
D=Path(os.environ.get('VESSEL_DATA_DIR','temp/vessel-noise')); D.mkdir(parents=True,exist_ok=True); A=D/'audio-prefixes'; A.mkdir(exist_ok=True)
if not (D/'vtuad-record.json').exists():
 r=requests.get('https://zenodo.org/api/records/23037895',timeout=40);r.raise_for_status();(D/'vtuad-record.json').write_text(json.dumps(r.json()))
record=json.loads((D/'vtuad-record.json').read_text()); files={f['key']:f for f in record['files']}
if not (D/'registry_anonymized.csv').exists():
 r=requests.get(files['registry_anonymized.csv']['links']['self'],timeout=40);r.raise_for_status();(D/'registry_anonymized.csv').write_bytes(r.content)
from catalog import catalog
if not (D/'strict-vessel-catalog.json').exists():(D/'strict-vessel-catalog.json').write_text(json.dumps(catalog('vtuad_strict_wav_vessel.zip.part03',509)))
with (D/'registry_anonymized.csv').open() as f: registry=list(csv.DictReader(f))
vessel=json.loads((D/'strict-vessel-catalog.json').read_text()); background=catalog('vtuad_strict_wav_background.zip',16)
entries={('/'.join(x['name'].split('/')[-2:])):x for x in vessel+background}
eligible=[x for x in registry if x['path'].removeprefix('./') in entries and entries[x['path'].removeprefix('./')]['uncompressed']>=7680044]
# First two available distinct hulls per class before September; validation uses later, unseen hulls.
selected=[]; train_hulls=set()
for label in sorted({x['label'] for x in eligible if x['label']!='background'}):
 pool=sorted([x for x in eligible if x['label']==label and x['date']<'20160901'],key=lambda x:hashlib.sha256(('192'+x['path']).encode()).hexdigest())
 n=0
 for x in pool:
  if x['hull_id'] in train_hulls:continue
  selected.append({**x,'split':'train'});train_hulls.add(x['hull_id']);n+=1
  if n==2:break
valid_hulls=set()
for label in sorted({x['label'] for x in eligible if x['label']!='background'}):
 pool=sorted([x for x in eligible if x['label']==label and x['date']>='20161001' and x['hull_id'] not in train_hulls|valid_hulls],key=lambda x:hashlib.sha256(('192'+x['path']).encode()).hexdigest())
 if pool:selected.append({**pool[0],'split':'heldout'});valid_hulls.add(pool[0]['hull_id'])
# Separate AIS-background recordings are an audit control, not confirmed vessel-free examples.
selected += [{**x,'split':'background_control'} for x in sorted([x for x in eligible if x['label']=='background'],key=lambda x:hashlib.sha256(('192'+x['path']).encode()).hexdigest())[:4]]
assert len(train_hulls)>=6 and len(valid_hulls)>=4 and train_hulls.isdisjoint(valid_hulls)
train_days={x['date'] for x in selected if x['split']=='train'}; val_days={x['date'] for x in selected if x['split']=='heldout'}
assert train_days.isdisjoint(val_days)
(D/'selected-references.json').write_text(json.dumps(selected,indent=2))
print('selection',len(selected),'train',len(train_hulls),'heldout',len(valid_hulls),'days',len(train_days),len(val_days),flush=True)
session=requests.Session()
def read_range(parts,start,end):
 chunks=[]; base=0
 for name in parts:
  info=files[name]; size=info['size']; lo=max(start,base); hi=min(end,base+size-1)
  if lo<=hi:
   r=session.get(info['links']['self'],headers={'Range':f'bytes={lo-base}-{hi-base}'},timeout=40)
   r.raise_for_status(); assert r.status_code==206 and len(r.content)==hi-lo+1 and r.headers.get('Content-Range','').startswith(f'bytes {lo-base}-{hi-base}/')
   chunks.append(r.content)
  base+=size
 data=b''.join(chunks); assert len(data)==end-start+1
 return data

def power(samples,fs):
 samples=samples.astype(np.float64); seconds=int(len(samples)/fs); mins=seconds//60
 segments=samples[:mins*60*fs].reshape(-1,2*fs)
 freq,psd=periodogram(segments,fs=fs,window='hann',detrend='constant',scaling='density',axis=1)
 centres=10.0**(1.5+.1*np.arange(25)); bands=[]
 for c in centres:
  mask=(freq>=c/10**.05)&(freq<c*10**.05)
  assert c*10**.05<=fs/2 and mask.any()
  bands.append(psd[:,mask].sum(axis=1)*(freq[1]-freq[0]))
 return np.array(bands).T.reshape(mins,30,25).mean(axis=1)

features=[]; audit=[]; began=time.monotonic()
for index,item in enumerate(selected):
 path=item['path'].removeprefix('./'); entry=entries[path]; stem=path.replace('/','_').replace('.wav',''); dest=A/(stem+'.wav')
 parts=['vtuad_strict_wav_background.zip'] if path.startswith('background/') else [f'vtuad_strict_wav_vessel.zip.part{i:02d}' for i in [1,2,3]]
 downloaded=0
 if not dest.exists():
  off=entry['header_offset']; head=read_range(parts,off,off+29); assert head[:4]==b'PK\x03\x04'; n,e=struct.unpack_from('<HH',head,26); start=off+30+n+e
  decoder=zlib.decompressobj(-15); raw=bytearray(); consumed=0
  while consumed<entry['compressed']:
   count=min(4*1024*1024,entry['compressed']-consumed); comp=read_range(parts,start+consumed,start+consumed+count-1); consumed+=count; downloaded+=count
   raw.extend(decoder.decompress(comp,16_000_000-len(raw)))
   with wave.open(io.BytesIO(raw)) as wav:
    fs=wav.getframerate(); channels=wav.getnchannels(); width=wav.getsampwidth(); assert channels==1 and width==2 and fs==32000
    samples=wav.readframes(120*fs)
   if len(samples)>=120*fs*width:break
  assert len(samples)==120*fs*width
  with wave.open(str(dest),'wb') as wav:
   wav.setnchannels(channels);wav.setsampwidth(width);wav.setframerate(fs);wav.writeframes(samples)
 with wave.open(str(dest),'rb') as wav:fs=wav.getframerate(); samples=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2')
 band=power(samples,fs); assert band.shape==(2,25) and np.isfinite(band).all() and (band>0).all()
 features.append(band)
 audit.append({'path':path,'label':item['label'],'hull_id':item['hull_id'],'date':item['date'],'split':item['split'],'prefix_seconds':120,'compressed_bytes_this_run':downloaded,'fs':fs})
 np.savez_compressed(D/'references-progress.npz',bands=np.asarray(features))
 (D/'reference-audit.json').write_text(json.dumps(audit,indent=2))
 print(index+1,len(selected),path,item['label'],item['split'],'MB',round(downloaded/1e6,2),'elapsed',round(time.monotonic()-began,1),flush=True)
np.savez_compressed(D/'references.npz',bands=np.asarray(features))
print('complete',len(features),'seconds',time.monotonic()-began,flush=True)
