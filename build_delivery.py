"""Package the selected completed run with explicit source/asset dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',required=True)
args=parser.parse_args()
run=ROOT/'outputs'/args.run
assessment=json.loads((run/'assessment.json').read_text())
if not assessment['outcome']['sampled_task_success']:
    raise RuntimeError('Refusing to label an unsuccessful run as the selected delivery')
config=json.loads((ROOT/'config.json').read_text())
if config!=json.loads((run/'config.json').read_text()):
    raise RuntimeError('Default configuration must match the selected run')
names={
    'README.md','requirements.txt','run.sh','runtime.py','simulate.py','config.json',
    'velcro.py','contact_friction.py','validation.py','assess_run.py','plot_metrics.py',
    'replay.py','test_validation.py','test_assess_run.py','test_velcro.py',
    'test_contact_friction.py','smoke_contact_friction.py','build_delivery.py',
    'planning/plan_outward_backup.py','planning/check_outward_backup_collisions.py',
    'planning/check_actual_collisions.py',
}
for directory in [ROOT/'assets',ROOT/'references',run]:
    for path in directory.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            names.add(str(path.relative_to(ROOT)))
for path in (ROOT/'planning').glob('outward_backup*'):
    if path.is_file():names.add(str(path.relative_to(ROOT)))
names.add(config['joint_trajectory'])
missing=[name for name in names if not (ROOT/name).is_file()]
if missing:raise FileNotFoundError(missing)
hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sorted(names)}
manifest={
    'selected_run':args.run,
    'input_scope':'Four provided initial images and the dressing task description; no original simulation files.',
    'physics':'Native Genesis 1.4.3 PBD plus disclosed sparse grasp/seam/contact-friction approximations.',
    'sampled_task_success':True,
    'final_qualifying_hold_seconds':assessment['outcome']['final_qualifying_hold_duration_s'],
    'included_files_sha256':hashes,
    'excluded':'Caches, failed/intermediate dynamic runs, and unrelated planning probes.',
}
manifest_path=ROOT/'delivery_manifest.json'
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
names.add('delivery_manifest.json')
target=ROOT/'four_view_genesis_reconstruction.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for name in sorted(names):
        archive.write(ROOT/name,'four_view_genesis/'+name)
with zipfile.ZipFile(target) as archive:
    bad=archive.testzip()
    if bad:raise RuntimeError(f'ZIP integrity failure: {bad}')
print(json.dumps({'archive':str(target),'bytes':target.stat().st_size,'files':len(names)},indent=2))
