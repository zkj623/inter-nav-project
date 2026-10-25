"""Create persistent movable-box overlays for all five original Go2 scenes."""
import argparse
from grutopia_extension.interactive_navigation.interactive_box_scenarios import prepare_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--scenes', default='mv7,mv7dq,mvucby,mvuh5ai,mvuhwaq')
    parser.add_argument('--source-layout-root', help='Preserve earlier positions and reuse their architectural grids')
    args = parser.parse_args()
    prepare_all(args.output, scenes=args.scenes.split(','), source_layout_root=args.source_layout_root)
