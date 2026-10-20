import os
"""Read public ZIP directory metadata; audio members are left untouched."""
import requests,json,struct,csv
from pathlib import Path
D=Path(os.environ.get('VESSEL_DATA_DIR','temp/vessel-noise')); j=json.loads((D/'vtuad-record.json').read_text())
def catalog(filename,count):
 f=next(x for x in j['files'] if x['key']==filename); cache=D/(filename+'.tail')
 if not cache.exists():
  r=requests.get(f['links']['self'],headers={'Range':'bytes=-262144'},timeout=30); r.raise_for_status(); assert r.status_code==206; cache.write_bytes(r.content)
 data=cache.read_bytes(); rows=[]; pos=data.find(b'PK\x01\x02')
 while pos>=0 and data[pos:pos+4]==b'PK\x01\x02':
  n,e,c=struct.unpack_from('<HHH',data,pos+28); comp=struct.unpack_from('<H',data,pos+10)[0]; compressed,uncompressed=struct.unpack_from('<II',data,pos+20); off=struct.unpack_from('<I',data,pos+42)[0]
  name=data[pos+46:pos+46+n].decode('utf-8'); extra=data[pos+46+n:pos+46+n+e]; k=0
  while k+4<=len(extra):
   tag,length=struct.unpack_from('<HH',extra,k); value=extra[k+4:k+4+length]; k+=4+length
   if tag==1:
    nums=list(struct.unpack('<'+'Q'*(len(value)//8),value[:len(value)//8*8]))
    if uncompressed==0xffffffff: uncompressed=nums.pop(0)
    if compressed==0xffffffff: compressed=nums.pop(0)
    if off==0xffffffff: off=nums.pop(0)
  assert max(compressed,uncompressed,off)!=0xffffffff and compressed>0
  rows.append(dict(name=name,compression=comp,compressed=compressed,uncompressed=uncompressed,header_offset=off)); pos+=46+n+e+c
 assert len(rows)==count and len({x['name'] for x in rows})==count
 return rows
if __name__=='__main__':
 rows=catalog('vtuad_strict_wav_vessel.zip.part03',509)
 (D/'strict-vessel-catalog.json').write_text(json.dumps(rows,indent=2))
 print('members',len(rows),'compressed_MB',sum(x['compressed'] for x in rows)/1e6)
 print('smallest',json.dumps(sorted(rows,key=lambda x:x['uncompressed'])[:6],indent=2))
 with (D/'registry_anonymized.csv').open() as q: reg=list(csv.DictReader(q))
 print('registry rows',len(reg),'keys',list(reg[0])); print('example',{k:v for k,v in reg[0].items() if k not in ['clip_uid','file_sha256','hull_id']})
