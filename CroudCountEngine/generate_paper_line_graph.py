"""Create a line graph from real full-frame and tiled-inference timing runs.

The figure is a configuration comparison, not an accuracy benchmark. It records
wall-clock detector time for the selected inputs under identical model settings.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from generate_paper_figures import build_detector


def run(detector, image):
    started = time.perf_counter()
    boxes, scores = detector.detect_tiled(image) if detector.tiled else detector.detect_image(image)
    if not detector.tiled:
        boxes, scores = detector.merge_boxes(boxes, scores)
    return (time.perf_counter() - started) * 1000, len(boxes)


def arguments(tiled: bool, base: argparse.Namespace) -> argparse.Namespace:
    values = vars(base).copy(); values['tiled'] = tiled
    return argparse.Namespace(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a real line graph comparing full-frame and tiled detector timing.')
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--output-dir', default='paper_figures')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--conf', type=float, default=.50); parser.add_argument('--iou', type=float, default=.45)
    parser.add_argument('--imgsz', type=int, default=640); parser.add_argument('--max-det', type=int, default=300)
    parser.add_argument('--tile-size', type=int, default=640); parser.add_argument('--tile-overlap', type=float, default=.20)
    parser.add_argument('--tile-batch-size', type=int, default=0); parser.add_argument('--cpu-threads', type=int, default=0); parser.add_argument('--device', default='cpu')
    parser.add_argument('--scales', default='1.0'); parser.add_argument('--tta-flip', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--count-scale', type=float, default=1.0); parser.add_argument('--count-offset', type=float, default=0.0); parser.add_argument('--roi', default='')
    parser.add_argument('--min-box-area', type=float, default=0.0); parser.add_argument('--max-box-area', type=float, default=0.0); parser.add_argument('--min-box-height', type=float, default=0.0)
    parser.add_argument('--max-box-width', type=float, default=0.0); parser.add_argument('--max-box-height', type=float, default=0.0); parser.add_argument('--min-aspect', type=float, default=0.0); parser.add_argument('--max-aspect', type=float, default=0.0)
    parser.add_argument('--center-merge', type=float, default=0.0); parser.add_argument('--center-merge-px', type=float, default=0.0)
    args=parser.parse_args(); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    full, tiled = build_detector(arguments(False,args)), build_detector(arguments(True,args))
    rows=[]
    for idx, source in enumerate(args.inputs,1):
        image=cv2.imread(source)
        if image is None: raise ValueError(f'Cannot read {source}')
        full_ms, full_count=run(full,image); tiled_ms, tiled_count=run(tiled,image)
        rows.append({'frame':idx,'source':Path(source).name,'full_frame_ms':round(full_ms,2),'tiled_ms':round(tiled_ms,2),'full_frame_count':full_count,'tiled_count':tiled_count})
    x=np.arange(1,len(rows)+1); full_values=[r['full_frame_ms']/1000 for r in rows]; tiled_values=[r['tiled_ms']/1000 for r in rows]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
    fig,ax=plt.subplots(figsize=(8.2,4.7),dpi=220); fig.patch.set_facecolor('white')
    box=FancyBboxPatch((.025,.05),.95,.90,boxstyle='round,pad=.012,rounding_size=.018',transform=fig.transFigure,facecolor='white',edgecolor='black',linewidth=.8,zorder=-1); fig.patches.append(box)
    ax.plot(x,full_values,color='#5B9BD5',marker='s',markersize=5,linewidth=1.8,label='Full-frame inference')
    ax.plot(x,tiled_values,color='#E41A1C',marker='o',markersize=5,linewidth=1.8,label='Tiled inference')
    ax.set_xticks(x,[f'{v}th' if v not in (1,2,3) else {1:'1st',2:'2nd',3:'3rd'}[v] for v in x]); ax.set_xlabel('Selected input frames',loc='right')
    ax.set_ylabel('Processing time (s)',loc='top',rotation=0,labelpad=15); ax.grid(axis='y',alpha=.35,linewidth=.7)
    ax.spines[['top','right','left','bottom']].set_visible(False); ax.tick_params(length=0); ax.legend(loc='lower center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False)
    ax.set_title('Detector Processing Time Across Selected Input Frames',fontsize=11,pad=12)
    fig.subplots_adjust(bottom=.25,left=.11,right=.95,top=.86)
    fig_path=out/'Fig_06_processing_time_comparison.png'; fig.savefig(fig_path,bbox_inches='tight'); plt.close(fig)
    with (out/'Fig_06_processing_time_comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(fig_path)

if __name__=='__main__': main()
