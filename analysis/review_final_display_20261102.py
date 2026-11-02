"""D's fixed E-v7/A-v4 content, identity and export-boundary review."""
from pathlib import Path
import hashlib
import importlib.util
import json
from unittest.mock import patch
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from pypdf import PdfReader

from .review_a_v3_statistics import numeric_tables, pareto_tables


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    e=Path('results/e_v7_review_20261102/deliverables/e_m5')
    old=Path('results/e_review_20261029/deliverables/e_m5')
    a=Path('results/a_v4_review_20261102')
    out=Path('analysis/dcu_20261102');out.mkdir(exist_ok=True)
    ppt=next(e.glob('*v7.pptx'));pdf=next(e.glob('*v7.pdf'))
    with ZipFile(ppt) as z:
        slides=sorted([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')],key=lambda n:int(n.split('slide')[-1][:-4]))
        assert len(slides)==12
        text=lambda n:' | '.join(ET.fromstring(z.read(n)).itertext())
        s12=text('ppt/slides/slide12.xml')
        notes=text('ppt/notesSlides/notesSlide4.xml')+'\n'+text('ppt/notesSlides/notesSlide12.xml')
        assert 'AUC 七项效应均显著' in s12 and '仅数据集主效应未显著' in s12
        assert '不是统一解释质量排名' in s12 and '全组计时仍待记录' in s12
        assert '.121215' in notes and '.1148' in notes and '9,999' in notes and '.0014' in notes
        assert 'IID' in notes and '.8374' in notes and '不是 p=0' in notes
        media={hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('ppt/media/')}
        assert sha(Path('analysis/dcu_20261028/method_model_profiles.png')) in media
        stale='v7 尚未 commit/push' in text('ppt/notesSlides/notesSlide12.xml')
    reader=PdfReader(pdf)
    assert len(reader.pages)==12
    demo_files=[p for p in (e/'demo').rglob('*') if p.is_file()]
    for p in demo_files:
        assert p.read_bytes()==(old/p.relative_to(e)).read_bytes(),str(p)
    report=(a/'docs/FINAL_REPORT_DRAFT.md').read_text(encoding='utf-8')
    old_report=Path('results/a_v3_review_20261101/docs/FINAL_REPORT_DRAFT.md').read_text(encoding='utf-8')
    tables=lambda s:[line for line in s.splitlines() if line.startswith('|')]
    assert tables(report)==tables(old_report)
    load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    counts=numeric_tables(report,load('analysis/dcu_20261028/repeated_measures.json'),load('analysis/dcu_20261029/mixed_factorial.json'))
    stats=load('analysis/dcu_20261015/statistics.json')
    assert len(pareto_tables(report,stats))==6
    # Source inspected before execution. Capture all axis-title bounds at export DPI.
    spec=importlib.util.spec_from_file_location('analysis._a_v4_verified_plot',a/'analysis/plot_final_pareto_3d.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from matplotlib.figure import Figure
    save=Figure.savefig;bounds=[]
    def capture(fig,*args,**kwargs):
        fig.canvas.draw();renderer=fig.canvas.get_renderer();box=kwargs['bbox_inches']
        for ax in fig.axes:
            for label in (ax.xaxis.label,ax.yaxis.label,ax.zaxis.label):
                b=label.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
                bounds.append(box.x0<=b.x0 and box.y0<=b.y0 and box.x1>=b.x1 and box.y1>=b.y1)
        return save(fig,*args,**kwargs)
    with patch.object(Figure,'savefig',capture):
        module.render(stats,Path('results/a_v4_review_20261102/verified_export.png'))
    assert len(bounds)==18 and all(bounds)
    receipt=dict(date='2026-11-02',actor='D',E_head='d00553f9ca7fc41208e24296619c1406a339d8da',
        A_head='3cfd4a8250b7f820e241bbcae3df93aa374f6c8a',pptx_sha256=sha(ppt),pdf_sha256=sha(pdf),
        pptx_pages=12,pdf_pages=12,pdf_searchable=any(p.extract_text() for p in reader.pages),
        demo_tracked_files_byte_identical_to_reviewed_v6=len(demo_files),stage_figure_byte_identity=True,
        statistical_content_checks='passed; new slide12 and notes4/12 compared with D results; both tests, finite resolution, task-score and DID limits retained',
        stale_speaker_note_found=stale,stale_location='ppt/notesSlides/notesSlide12.xml',
        report_table_lines_unchanged=len(tables(report)),**counts,A_fourteen_quality_rows_rechecked=True,
        A_pareto_flags_rechecked=36,A_axis_titles_inside_export_bounds=int(sum(bounds)),
        reviewed_plot_code_sha256=sha(a/'analysis/plot_final_pareto_3d.py'),
        scope='D content/identity and local rendering review only; no PowerPoint application, independent-device demo, full bootstrap or actual timed rehearsal',
        review_code_sha256=sha(Path(__file__)))
    (out/'final_display_review_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':
    main()
