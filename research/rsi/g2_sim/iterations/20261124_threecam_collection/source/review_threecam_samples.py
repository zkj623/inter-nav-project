"""Render accepted samples for review without changing or promoting data."""
from pathlib import Path
import argparse, json
import h5py
import numpy as np
from PIL import Image, ImageDraw
R = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--all', action='store_true', help='Render all accepted samples; default first per object')
a = p.parse_args()
O = R / 'threecam_review'
O.mkdir(exist_ok=True)
index = []
for statefile in sorted((R / 'collection_state').glob('*.json')):
    state = json.loads(statefile.read_text())
    accepted = state['accepted'] if a.all else state['accepted'][:1]
    for row in accepted:
        T = Path(row['trial'])
        canvas = Image.new('RGB', (960, 616), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"{row['asset']} p{row['index']:04d}: dx={1000*row['dx_m']:.2f} mm, dy={1000*row['dy_m']:.2f} mm", fill='black')
        with h5py.File(T / 'episode.hdf5') as f:
            n = len(f['action'])
            # Initial common home and midpoint of final 3-second stable hold.
            indices = [0, max(0, n - 30)]
            for r, t in enumerate(indices):
                for c, name in enumerate(['head_front', 'wrist_left', 'wrist_right']):
                    im = f['observations/images/' + name][t]
                    if name == 'head_front':
                        im = np.pad(im, ((64, 64), (0, 0), (0, 0)))
                    tile = Image.fromarray(im).resize((320, 264))
                    y = 48 + r * 284
                    canvas.paste(tile, (c * 320, y))
                    draw.text((c * 320 + 5, y - 17), f'{name}: frame {t}', fill='black')
        name = f"{row['asset']}_p{row['index']:04d}.jpg"
        canvas.save(O / name, quality=90)
        index.append(dict(asset=row['asset'], placement_index=row['index'], trial=str(T), image=name))
(O / 'index.json').write_text(json.dumps(index, indent=2))
print(json.dumps(dict(samples_rendered=len(index), directory=str(O))))
