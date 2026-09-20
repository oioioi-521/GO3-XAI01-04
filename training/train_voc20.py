"""Reproducible VOC20 multi-label trainer; frozen metadata is never used as training data."""
from __future__ import annotations
import argparse, csv, hashlib, json, time, sys
from pathlib import Path
import torch, yaml
from torch.utils.data import DataLoader, Dataset
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from preprocessing.dataset import load_image
from preprocessing.extract_voc_labels import VOC_CLASSES
from models.factory import load_model
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
class Manifest(Dataset):
 def __init__(self,path):
  with Path(path).open(encoding='utf8',newline='') as f:self.rows=list(csv.DictReader(f))
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; return load_image(r['image_path']),torch.tensor(json.loads(r['targets']),dtype=torch.float32),r['image_id']
def run(path,smoke=False):
 c=yaml.safe_load(Path(path).read_text()); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed']); dev=torch.device('cuda'); tr=Manifest(ROOT/c['data']['train_manifest']); va=Manifest(ROOT/c['data']['validation_manifest'])
 assert not set(r['image_id'] for r in tr.rows)&set(r['image_id'] for r in va.rows)
 counts=torch.tensor([sum(json.loads(r['targets'])[i] for r in tr.rows) for i in range(20)],device=dev); weight=((len(tr)-counts)/counts).clamp(max=20)
 model=load_model(c['model']['name'],dev,'default',20).train(); opt=torch.optim.SGD(model.parameters(),lr=c['optimizer']['lr'],momentum=.9,weight_decay=1e-4); lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=weight); scaler=torch.amp.GradScaler('cuda'); best=-1; ck=ROOT/c['output']['checkpoint']; ck.parent.mkdir(parents=True,exist_ok=True)
 for epoch in range(1,(1 if smoke else c['epochs'])+1):
  for x,y,_ in DataLoader(tr,batch_size=c['runtime']['batch_size'],shuffle=True):
   x,y=x.to(dev),y.to(dev);opt.zero_grad(set_to_none=True)
   with torch.amp.autocast('cuda'):loss=lossfn(model(x),y)
   scaler.scale(loss).backward();scaler.step(opt);scaler.update()
  model.eval(); logits=[];targets=[]
  with torch.inference_mode():
   for x,y,_ in DataLoader(va,batch_size=c['runtime']['batch_size']):logits.append(model(x.to(dev)).cpu());targets.append(y)
  p=torch.cat(logits).sigmoid();t=torch.cat(targets); ap=[]
  for i in range(20):
   o=torch.argsort(p[:,i],descending=True);z=t[:,i][o];ap.append(float((z*z.cumsum(0)/torch.arange(1,len(z)+1)).sum()/max(1,z.sum())))
  report={'mAP':sum(ap)/20,'per_class_ap':dict(zip(VOC_CLASSES,ap))}
  if report['mAP']>best: best=report['mAP'];torch.save({'state_dict':model.state_dict(),'VOC_CLASSES':VOC_CLASSES,'task_type':'multilabel','output_activation':'sigmoid','manifest_hash':sha(ROOT/c['data']['train_manifest']),'best_epoch':epoch,'validation':report},ck)
 record={'smoke':smoke,'best_epoch':epoch,'validation':report,'checkpoint_sha256':sha(ck),'frozen_eval_used':False};out=ROOT/c['output']['record'];out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,ensure_ascii=False,indent=2));print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();run(a.config,a.smoke)
