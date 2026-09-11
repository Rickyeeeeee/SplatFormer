"""Analyze existing overfit logs without loading checkpoints or modifying runs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import statistics
os.environ.setdefault('MPLCONFIGDIR','/tmp/ptv3_report_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path('/project2/ricky/experiments')
KEYS=['means_loss','features_dc_loss','features_rest_loss','opacities_loss','scales_loss','quats_loss']
run_paths=sorted((ROOT/'0910/overfit_sr_gsfm_512_input_frame_v1').glob('*grid2048_scenes*/overfit.log'))
run_paths+=sorted((ROOT/'0911/overfit_sr_gsfm_512').glob('many_19_*grid*_batch_size*/overfit.log'))
records=[]
for log_path in run_paths:
    path=log_path.parent
    log=log_path.read_text()
    config=(path/'config.gin').read_text()
    selection=json.loads((path/'selected_scenes.json').read_text())
    rows=[]
    # Some log files append multiple attempts: keep the final monotonic training segment.
    for line in log.splitlines():
        if re.search(r'\bstep=\d+ .*\btotal=',line):
            row={k:float(v) for k,v in re.findall(r'\b(\w+)=(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)',line)}
            if rows and row['step']<=rows[-1]['step']: rows=[]
            rows.append(row)
    if not rows: continue
    n=len(selection['scenes'])
    batch=int(rows[-1].get('batch_size',1))
    steps=int(re.search(r'training.total_steps = (\d+)',config)[1])
    metrics={}
    for f in sorted((path/'eval_final').glob('flow_steps_*/metrics.json')):
        d=json.loads(f.read_text())
        if 'mean' in d: metrics[f.parent.name]=d
    baseline_path=path/'baseline_render_metrics.json'
    baseline=json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    snapshot={'path':str(path),'date':path.parts[4], 'scene_count':n,'batch_size':batch,'planned_steps':steps,'last_logged_step':int(rows[-1]['step']), 'nominal_presentations_per_scene':steps*batch/n,'grid':int(re.search(r'GSFlowPredictor.grid_resolution = (\d+)',config)[1]),'variance_source':re.search(r"flow_matching.velocity_variance_source = '([^']+)'",config)[1], 'tail_mean':{k:statistics.mean(r[k] for r in rows[-100:]) for k in ['total']+KEYS},'tail_logged_rows':min(100,len(rows)),'rows':rows,'final_metrics':metrics,'baseline':baseline,'selected_scenes':selection,'log_sha256':hashlib.sha256(log.encode()).hexdigest(),'config_sha256':hashlib.sha256(config.encode()).hexdigest()}
    records.append(snapshot)
selected={b:next(r for r in records if '/0911/' in r['path'] and r['grid']==2048 and r['batch_size']==b) for b in [1,4]}
assert selected[1]['config_sha256']==selected[4]['config_sha256']
assert selected[1]['selected_scenes']==selected[4]['selected_scenes']
assert selected[1]['baseline']==selected[4]['baseline']
summary={'snapshot_utc':datetime.now(timezone.utc).isoformat(),'method':'Existing logs only. Final monotonic training segment retained. Tail means use last 100 logged batches, not full-dataset evaluation. Each scene presentation contains one sampled time and all scene Gaussians. No new training.','records':records}
(OUT/'run_evidence.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
colors={1:'#176B87',4:'#D97242'}
fig,axs=plt.subplots(1,3,figsize=(17,5.5))
for b,r in selected.items():
    rows=r['rows']
    x=np.array([v['step'] for v in rows]); y=np.array([v['total'] for v in rows])
    smooth=np.convolve(y,np.ones(25)/25,mode='valid'); x=x[24:]
    axs[0].plot(x,smooth,label=f'Batch {b}',color=colors[b],lw=2)
    axs[1].plot((x+1)*b/19,smooth,label=f'Batch {b}',color=colors[b],lw=2)
    vals=[r['tail_mean'][k] for k in KEYS]
    axs[2].bar(np.arange(6)+(b==4)*.38,vals,width=.38,label=f'Batch {b}',color=colors[b])
axs[0].set_xlabel('Optimizer-loop steps'); axs[1].set_xlabel('Scene presentations per scene (average)')
for ax in axs[:2]:
    ax.set_yscale('log');ax.set_ylabel('Logged mean normalized velocity loss');ax.legend();ax.grid(alpha=.2)
axs[0].set_title('Same 20,000-step budget',fontweight='bold');axs[1].set_title('Exposure changes with batch size',fontweight='bold')
axs[2].set_xticks(np.arange(6)+.19);axs[2].set_xticklabels(['means','DC','SH-rest','opacity','scales','quats'],rotation=35,ha='right')
axs[2].set_ylabel('Mean contribution, last 100 logged batches');axs[2].legend();axs[2].set_title('SH-rest dominates the residual',fontweight='bold')
fig.suptitle('19-scene overfit: unchanged saved Gin configuration, different scene batches',fontsize=17,fontweight='bold')
fig.tight_layout(rect=(0,.035,1,.94))
fig.text(.015,.012,'Curves: trailing 25 logged batches (about 500 steps). Exposure counts are not equivalent optimizer updates; batch losses average scenes.',fontsize=10)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'training_diagnosis.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

r=selected[4];base=r['baseline']['scenes']
b1=selected[1]['final_metrics']['flow_steps_10']['scenes'];b4=r['final_metrics']['flow_steps_10']['scenes']
names=sorted(b4,key=lambda s:base[s]['matching_target_gs']['psnr']-b4[s]['psnr'],reverse=True)
fig,ax=plt.subplots(figsize=(15,7))
x=np.arange(len(names))
for label,values,color,marker in [('Source',[base[s]['source_gs']['psnr'] for s in names],'#a6b3b7','.'),('Batch 1',[b1[s]['psnr'] for s in names],colors[1],'o'),('Batch 4',[b4[s]['psnr'] for s in names],colors[4],'o'),('Matching target',[base[s]['matching_target_gs']['psnr'] for s in names],'#438C66','D')]:
    ax.plot(x,values,label=label,color=color,marker=marker,lw=1)
ax.set_xticks(x);ax.set_xticklabels([s[:8] for s in names],rotation=50,ha='right')
ax.set_ylabel('Mean rendered PSNR across HR views (dB)');ax.set_xlabel('Scene ID prefix; ordered by matching-target minus batch-4 PSNR')
ax.set_title('The remaining gap is to an existing same-count matching target',loc='left',fontweight='bold',pad=18)
ax.legend(ncol=4);ax.grid(axis='y',alpha=.2)
fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'per_scene_gap.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
rows='| Run | Steps | Batch | Presentations / scene | Tail velocity loss |\n|---|---:|---:|---:|---:|\n'
for r in records:
    if '/0911/' in r['path'] and r['grid']==1024: continue
    rows+=f"| {r['scene_count']} scenes ({'Sep 11' if '/0911/' in r['path'] else 'Sep 10'}) | {r['planned_steps']:,} | {r['batch_size']} | {r['nominal_presentations_per_scene']:,.0f} | {r['tail_mean']['total']:.4f} |\n"
(OUT/'run_table.md').write_text(rows)
print(rows)
print('Batch 4 SH-rest share:',selected[4]['tail_mean']['features_rest_loss']/selected[4]['tail_mean']['total'])
print('Paired PSNR improvement range:',min(b4[s]['psnr']-b1[s]['psnr'] for s in names), max(b4[s]['psnr']-b1[s]['psnr'] for s in names))
print('All scenes improved:',all(b4[s]['psnr']>b1[s]['psnr'] for s in names))
