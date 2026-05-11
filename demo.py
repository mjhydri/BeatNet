import numpy as np
import os.path as osp
import librosa

from argparse import ArgumentParser
from BeatNet.BeatNet import BeatNet

try:
    import moviepy.editor as mp
except ImportError:
    raise ImportError('Please install moviepy to enable music embedded with beats.')


def parse_args():
    parser = ArgumentParser(description='beat predictor')
    parser.add_argument('audio', type=str, help='audio file')
    parser.add_argument('--model',
                        default=1,
                        type=int,
                        choices=[1, 2, 3],
                        help='pre-trained model to utilize')
    parser.add_argument('--inference-model',
                        default='DBN',
                        choices=['DBN', 'PF'],
                        help='inference approachs')
    parser.add_argument('--mode',
                        default='offline',
                        choices=['offline', 'stream', 'online', 'realtime'],
                        help='inference mode')
    parser.add_argument('--plot',
                        default=None,
                        choices=['activations', 'beat_particles', 'downbeat_particles'],
                        help='different types of plots')
    parser.add_argument('--threading',
                        default=False,
                        action='store_true',
                        help='whether to use threading')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='device')
    parser.add_argument('--embed',
                        action='store_true',
                        help=('whether to embed the beats into the music. '
                              'Otherwise it saves them as np array.'))
    args = parser.parse_args()
    return args


def embed_to_music(args, output):
    video = mp.ColorClip((480, 480), (0, 0, 0), duration=output[-1][0])
    video.audio = mp.AudioFileClip(args.audio)
    subs = [row[1] for row in output]
    ts = [row[0] for row in output]
    clips = []
    for i in range(len(subs) - 1):
        text_clip = mp.TextClip(str(subs[i]), fontsize=40, color='white')
        text_clip = text_clip.set_start(ts[i])
        duration = ts[i+1] - ts[i]
        text_clip = text_clip.set_pos('center').set_duration(duration)
        clips.append(text_clip)
    clips.insert(0, video)
    result = mp.CompositeVideoClip(clips)
    result.write_videofile(f'{osp.splitext(args.audio)[0]}_{args.model}_beats.mp4', fps=25)

def main():
    args = parse_args()
    estimator = BeatNet(args.model,
                        mode=args.mode,
                        inference_model=args.inference_model,
                        plot=[] if args.plot is None else [args.plot],
                        thread=args.threading, device=args.device)

    print('\nPerforming beat recognition...')
    audio, _ = librosa.load(args.audio)
    output = estimator.process(audio)

    if not args.embed:
        out_file = osp.splitext(args.audio)[0]+'.npy'
        np.save(out_file, output)
        print(f'\nStored {out_file}')
    else:
        print(f'\nEmbedding the beats to the music...')
        embed_to_music(args, output)

if __name__ == '__main__':
    main()
