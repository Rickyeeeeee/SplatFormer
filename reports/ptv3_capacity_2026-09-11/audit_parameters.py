"""Reproduce the CPU parameter audit and figures; does not run training or inference."""
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('MPLCONFIGDIR', '/tmp/ptv3_report_matplotlib')
import gin
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from models.feature_flow_predictor import GSFlowPredictor
from models.pointtransformer_v3_flow import PointTransformerV3


def count(module):
    return sum(p.numel() for p in module.parameters())


def category(name):
    if name.startswith('features_outputhead.'):
        return 'Output heads'
    if '.t_mlp.' in name or '.fc_t' in name:
        return 'Time conditioning'
    if '.cpe.0.' in name:
        return '3D convolution in CPE'
    if '.cpe.' in name:
        return 'CPE linear + norm'
    if '.attn.' in name:
        return 'Attention QKV + projection'
    if '.mlp.' in name:
        return 'Block feed-forward MLP'
    if '.norm1.' in name or '.norm2.' in name:
        return 'Attention/MLP norms'
    return 'Stem + pool/unpool'


# Only the unrelated legacy FeaturePredictor binding is skipped; all GSFlow bindings resolve.
gin.parse_config_file('configs/model/ptv3_flow.gin', skip_unknown=['FeaturePredictor'])
torch.manual_seed(0)
model = GSFlowPredictor()
backbone = model.backbone.backbone
rows = [{'name': n, 'shape': list(p.shape), 'parameters': p.numel(), 'trainable': p.requires_grad, 'category': category(n)} for n, p in model.named_parameters()]
categories = {}
for row in rows:
    categories[row['category']] = categories.get(row['category'], 0) + row['parameters']
stages = {'Stem': count(backbone.embedding), 'Shared time MLP': count(backbone.fc_t1) + count(backbone.fc_t2)}
stages.update({name: count(stage) for name, stage in backbone.enc.named_children()})
stages.update({name: count(stage) for name, stage in backbone.dec.named_children()})
stages['Output heads'] = count(model.features_outputhead)
enc = [64, 96, 128, 256, 512]
dec = [96, 96, 128, 256]
e_depth = [2, 2, 2, 6, 2]
d_depth = [2, 2, 2, 2]
# Full 3x3x3 CPE + pointwise linear + attention + 4x FFN + three affine LNs + time linear.
block_formula = lambda c: 40*c*c + 21*c
formula = 24*64 + 87 + sum(d*block_formula(c) for c, d in zip(enc, e_depth))
formula += sum((enc[i-1]+1)*enc[i] for i in range(1, 5))
formula += sum(d*block_formula(c) for c, d in zip(dec, d_depth))
formula += sum((list(dec)[i+1] if i<3 else enc[-1])*dec[i] + enc[i]*dec[i] + 2*dec[i] for i in range(4))
head_formula = 6*((119+1)*128 + 2*(128+1)*128) + (128+1)*23
assert formula == count(model.backbone)
assert head_formula == count(model.features_outputhead)
assert sum(stages.values()) == sum(categories.values()) == count(model)
assert all(count(block) == block_formula(block.channels) for path in [backbone.enc, backbone.dec] for stage in path.children() for name, block in stage.named_children() if name.startswith('block'))

variants = []
settings = [
    ('Current flow', {}),
    ('Heads 256', {'GSFlowPredictor.output_head_width':256}),
    ('Decoder output 128', {'PointTransformerV3FlowModel.output_dim':128}),
    ('Twice decoder depth', {'PointTransformerV3FlowModel.dec_depths':(4,4,4,4)}),
    ('2 extra bottleneck blocks', {'PointTransformerV3FlowModel.enc_depths':(2,2,2,6,4)}),
    ('Time width 128 (count only)', {'PointTransformerV3FlowModel.T_dim':128}),
    ('All widths x1.5', {'PointTransformerV3FlowModel.enc_channels':(96,144,192,384,768), 'PointTransformerV3FlowModel.dec_channels':(144,144,192,384)}),
    ('BN enabled', {'PointTransformerV3FlowModel.turn_off_bn':False}),
    ('Time disabled', {'PointTransformerV3FlowModel.T_dim':-1}),
]
for label, bindings in settings:
    gin.clear_config()
    gin.parse_config_file('configs/model/ptv3_flow.gin', skip_unknown=['FeaturePredictor'])
    for key, value in bindings.items():
        gin.bind_parameter(key, value)
    variant = GSFlowPredictor()
    variants.append({'name':label, 'parameters':count(variant), 'delta':count(variant)-count(model), 'bindings':bindings})
    del variant
files = ['models/feature_flow_predictor.py', 'models/pointtransformer_v3_flow.py', 'Pointcept/pointcept/models/point_transformer_v3/point_transformer_v3m1_time.py', 'configs/model/ptv3_flow.gin', 'sr/flow.py', 'sr/alignment.py', 'train-sr-gsfm.py', 'overfit-sr-gsfm.py', 'scripts/train-sr-gsfm-on-objaverse.sh', 'scripts/overfit-sr-gsfm-on-objaverse.sh', 'models/POINTTRANSFORMER_V3_ARCHITECTURE.md']
result = {'total':count(model), 'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad), 'backbone':count(model.backbone), 'heads':count(model.features_outputhead), 'formula_total':formula+head_formula, 'stages':stages, 'categories':categories, 'heads_by_attribute':{n:count(h) for n,h in model.features_outputhead.items()}, 'variants':variants, 'parameter_tensors':len(rows), 'buffer_elements':sum(b.numel() for b in model.buffers()), 'source_class':inspect.getfile(PointTransformerV3), 'torch_version':torch.__version__, 'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}, 'validation':'Real CPU constructor; all trainable tensors counted; independent per-block and total formulas asserted. No forward, checkpoint, data, or training evaluation.'}
(OUT/'parameter_audit.json').write_text(json.dumps(result, indent=2)+'\n')
with (OUT/'parameters.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
fig, axes = plt.subplots(1,2,figsize=(16,7.2), gridspec_kw={'width_ratios':[1,1.35]})
items = [(n,v) for n,v in stages.items() if n.startswith(('enc','dec'))]
axes[0].barh([n for n,v in items], [v/1e6 for n,v in items], color=['#176B87' if n.startswith('enc') else '#58A399' for n,v in items])
axes[0].invert_yaxis()
for i,(n,v) in enumerate(items): axes[0].text(v/1e6+.15,i,f'{v/1e6:.3f}M',va='center',fontsize=10)
axes[0].set_xlim(0,25)
axes[0].set_title('Parameters by U-Net stage',loc='left',fontweight='bold')
axes[0].set_xlabel('Million trainable parameters (including transitions)')
items = sorted(categories.items(), key=lambda x:-x[1])
axes[1].barh([n for n,v in items], [v/1e6 for n,v in items],color=['#D97242','#176B87','#58A399','#8D80AD','#9DADB8','#BDCCD5','#BDCCD5','#BDCCD5'])
axes[1].invert_yaxis()
for i,(n,v) in enumerate(items): axes[1].text(v/1e6+.2,i,f'{v/1e6:.3f}M  |  {v/count(model):.2%}',va='center',fontsize=10)
axes[1].set_xlim(0,41)
axes[1].set_xlabel('Million trainable parameters')
axes[1].set_title('Where the 47.956M parameters go',loc='left',fontweight='bold')
fig.suptitle('GSFlowPredictor · measured from the configured model',fontsize=19,fontweight='bold',y=.99)
fig.tight_layout(pad=2.4)
for ext in ['png','svg','pdf']: fig.savefig(OUT/f'parameter_breakdown.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(13,5.5))
shown=variants[:7]
ax.barh([v['name'] for v in shown],[v['parameters']/1e6 for v in shown],color=['#176B87']+['#58A399']*5+['#D97242'])
ax.invert_yaxis()
for i,v in enumerate(shown): ax.text(v['parameters']/1e6+.8,i,f"{v['parameters']/1e6:.3f}M  ({v['delta']/1e6:+.3f}M)",va='center')
ax.set_xlim(0,138)
ax.set_xlabel('Million parameters; each variant changes only the listed setting')
ax.set_title('Capacity changes: exact construction counts, no quality measurements',loc='left',fontweight='bold',pad=15)
fig.text(.02,.01,'Time width 128 also needs a compatible input time encoder before a forward pass; it is not a working Gin-only change.',fontsize=10)
fig.tight_layout(rect=(0,.04,1,1))
for ext in ['png','svg','pdf']: fig.savefig(OUT/f'capacity_variants.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(16,10))
ax.set_xlim(0,16); ax.set_ylim(0,10); ax.axis('off')

def box(x,y,w,h,text,color='#EAF2F6',size=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.10',facecolor=color,edgecolor='#A7BBC6',linewidth=1.2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,color='#183442')


def arrow(a,b,color='#466C7C',style='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.6,'linestyle':style})

ax.text(.2,9.65,'Point Transformer v3 flow model: a time-conditioned sparse U-Net',fontsize=20,fontweight='bold',color='#183442')
box(.3,8.2,3.7,1,'Current Gaussian state x(t)\nN × 23 attributes\nmeans 3 · scales 3 · opacity 1\nquaternion 4 · SH color 12',size=10)
box(4.7,8.2,4.1,1,'Fixed source coordinates\nfloor(reference means × grid resolution)\nserialize four space-filling orders\npatch limit: 1,024 points',size=10)
box(10.1,8.2,5.5,1,'t → [t, sin(t), cos(t)] → MLP 3→12→3\nPer-block Linear(3, C), added after CPE\n22 block projections; 17,239 total time parameters',color='#FBEBD9',size=10)
arrow((2.15,8.2),(2.15,7.55)); arrow((6.75,8.2),(4.2,7.1))
box(.3,6.7,3.9,.85,'Stem Linear 23→64\nNo BatchNorm; GELU',size=11)
ys=[5.7,4.5,3.3,2.1,.9]
for i,(c,d,y) in enumerate(zip(enc,e_depth,ys)):
    n='N' if i==0 else f'N{i}'
    box(.3,y,3.9,.8,f'Encoder {i} · {n} × {c} · {d} blocks\n{stages[f"enc{i}"]/1e6:.3f}M parameters')
    if i==0: arrow((2.25,6.7),(2.25,y+.8))
    else:
        arrow((2.25,ys[i-1]),(2.25,y+.8))
        ax.text(2.5,y+.97,f'pool stride {[1,2,2,2][i-1]}',fontsize=9)
for i in range(4):
    y=ys[i]; c=dec[i]
    box(6.15,y,4.25,.8,f'Decoder {i} · {"N" if i==0 else "N"+str(i)} × {c} · 2 blocks\n{stages[f"dec{i}"]/1e6:.3f}M parameters',color='#E4F3EE')
    arrow((4.2,y+.4),(6.15,y+.4),style='--')
    if i<3: arrow((8.25,ys[i+1]+.8),(8.25,y))
arrow((4.2,1.3),(8.25,1.3)); arrow((8.25,1.3),(8.25,2.1))
box(6.15,6.7,4.25,.85,'Concat 96 + input 23 = 119\n6 heads: 119→128→128→128→d',color='#E4F3EE',size=10)
arrow((8.25,6.5),(8.25,6.7))
box(11.2,6.7,4.4,.85,'Velocity v(t): N × 23\nEuler: x ← x + v / steps',color='#FBEBD9')
arrow((10.4,7.1),(11.2,7.1))
box(11.2,3.35,4.4,2.7,'Each block at width C\n\n3³ sparse conv → Linear → LN + skip\n+ projected time embedding\nLN → local attention + skip\nLN → MLP C→4C→C + skip\n\nParameters: 40C² + 21C',size=10)
box(11.2,1.0,4.4,1.75,'Interpretation\n\n14 encoder + 8 decoder blocks\nUnpooling restores existing points\nNo point birth / split module\nN1…N4 depend on voxel occupancy',color='#FBEBD9',size=10)
ax.text(.3,.2,'Dashed arrows: projected encoder skip features; decoder adds gathered coarse features. Neighborhood coordinates stay tied to the source during sampling.',fontsize=10,color='#466C7C')
for ext in ['png','svg','pdf']: fig.savefig(OUT/f'architecture.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
print(json.dumps(result, indent=2))
