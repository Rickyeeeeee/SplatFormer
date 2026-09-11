"""Fill audited tables and build a self-contained, printable HTML report."""
import base64
import json
from pathlib import Path
import re
import markdown

OUT=Path(__file__).resolve().parent
data=json.loads((OUT/'parameter_audit.json').read_text())
total=data['total']
stage_table='| Component | Parameters | Share |\n|---|---:|---:|\n'+'\n'.join(f'| {k} | {v:,} | {v/total:.2%} |' for k,v in data['stages'].items())+f'\n| **Total** | **{total:,}** | **100%** |'
category_table='| Parameter family | Parameters | Share |\n|---|---:|---:|\n'+'\n'.join(f'| {k} | {v:,} | {v/total:.3%} |' for k,v in sorted(data['categories'].items(),key=lambda x:-x[1]))
variant_table='| Construction | Parameters | Change |\n|---|---:|---:|\n'+'\n'.join(f"| {v['name']} | {v['parameters']:,} | {v['delta']:+,} |" for v in data['variants'][:7])
parts=[]
for name in ['architecture_and_parameters','generation_assessment']:
    text=(OUT/f'{name}.md').read_text()
    for key,value in [('STAGE_TABLE',stage_table),('CATEGORY_TABLE',category_table),('VARIANT_TABLE',variant_table)]:
        text=text.replace(f'<!-- {key} -->',f'<!-- {key}_START -->\n{value}\n<!-- {key}_END -->')
        text=re.sub(f'<!-- {key}_START -->.*?<!-- {key}_END -->',lambda m:f'<!-- {key}_START -->\n{value}\n<!-- {key}_END -->',text,flags=re.S)
    (OUT/f'{name}.md').write_text(text)
    body=markdown.markdown(text,extensions=['tables','fenced_code','toc'])
    for file in OUT.glob('*.png'):
        uri='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode()
        body=body.replace(f'src="{file.name}"',f'src="{uri}"')
    body=body.replace('href="architecture_and_parameters.md"','href="#architecture"').replace('href="generation_assessment.md"','href="#assessment"')
    parts.append(body)
css='''body{font:17px/1.65 system-ui,sans-serif;color:#193442;margin:0;background:#eef3f5}main{max-width:1150px;margin:0 auto;padding:45px 55px;background:white}h1{font-size:2.1em;line-height:1.18;margin-top:1.2em}h2{margin-top:2em;border-bottom:2px solid #e2ecef;padding-bottom:8px}h3{margin-top:1.8em}a{color:#176b87}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:.88em;margin:24px 0}th{background:#e9f2f5;text-align:left}td,th{padding:9px 12px;border-bottom:1px solid #d9e4e8}tr:nth-child(even){background:#f7fafb}pre{padding:18px;background:#eff4f6;overflow:auto}code{font-size:.88em}nav{display:flex;gap:25px;flex-wrap:wrap;padding:14px 20px;background:#183442}nav a{color:white}section+section{border-top:5px solid #58a399;margin-top:70px;padding-top:30px}.meta{font-size:14px;color:#55717d}button{cursor:pointer;padding:8px 15px}@media(max-width:700px){main{padding:20px}table{font-size:.73em}td,th{padding:5px}h1{font-size:1.7em}}@media print{body{background:white}main{max-width:none;padding:0;font-size:10pt}nav,button{display:none}img,table{break-inside:avoid}h2,h3{break-after:avoid}section+section{break-before:page;border:0}a{color:#193442}h1{font-size:23pt}}'''
html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PTv3 architecture, parameters and generation assessment</title><style>'+css+'</style></head><body><main><nav><a href="#architecture">Architecture & parameters</a><a href="#assessment">Generation assessment</a><a href="parameter_audit.json">Audit data</a></nav><p class="meta">SplatFormer · source audit · 11 September 2026 · CPU construction counts, no quality benchmark</p><button onclick="window.print()">Print / save as PDF</button><section id="architecture">'+parts[0]+'</section><section id="assessment">'+parts[1]+'</section></main></body></html>'
(OUT/'report.html').write_text(html)
print('Generated report.html and populated report tables.')
