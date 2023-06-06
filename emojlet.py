from image_similarity_measures.quality_metrics import metric_functions

import cv2
import numpy as np

import os
import sys
import argparse

from math import ceil

parser = argparse.ArgumentParser(
  prog="emojlet",
  description="render image as emoji tiles",
)
parser.add_argument('tiles_dir', help="directory with emoji image files")
parser.add_argument('in_image', help="input image")
parser.add_argument('--tile', dest='tile', help="tile width", default='18')
parser.add_argument('--patch', dest='patch', help="patch width", default='64,78')
parser.add_argument('--bg', dest='bg_color',
                    help="fill transparent areas with RRGGBB value", default=None)

args = parser.parse_args()

def dims_from_spec(dimspec):
  dims = tuple(int(dim) for dim in dimspec.split(','))
  if len(dims)==1:
    dims = (dims[0], dims[0])
  return dims

tile_size = dims_from_spec(args.tile)
patch_size = dims_from_spec(args.patch)
if args.bg_color:
  rgb_bgcolor = tuple(int(args.bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
  bgr_bgcolor = (rgb_bgcolor[2], rgb_bgcolor[1], rgb_bgcolor[0])

def solid_background(foreground, bgcolor):
    background = np.full(foreground.shape[:2] + (3,), bgcolor, np.uint8)
    # separate fg alpha & color channels
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine background with alpha-weighted overlay
    return background * (1 - alpha_mask) + foreground_colors * alpha_mask

ecabulary = {}
filenames = os.listdir(args.tiles_dir)

for f in sorted(filenames):
    if not f.endswith(".png"):
      continue
    try:
      fname, ext = f.split('.')
      undersplits = fname.split('_')
      if len(undersplits) == 2:
        emoji_chars = undersplits[-1]
      elif len(undersplits) == 4:
        emoji_chars = undersplits[-2]
      else:
        print(f"ERROR: unparseable filename {f}")
        continue
      emoji_zwjseq = ''.join(chr(int(hexstr, 16)) for hexstr in emoji_chars.split("-"))
      if emoji_zwjseq == u'\uFE0F':
        continue
      if emoji_zwjseq in ecabulary:
        print(f"DUPLICATE: {emoji_zwjseq} {f}")
        continue
      emoji_tile = cv2.imread(os.path.join(sys.argv[1], f), cv2.IMREAD_UNCHANGED)  # TODO cv.IMREAD_UNCHANGED to save transparency
      if args.bg_color:
        emoji_tile = solid_background(emoji_tile, bgr_bgcolor)
      ecabulary[emoji_zwjseq] = (f, cv2.resize(emoji_tile, tile_size))
    except Exception as e:
      print(f"{f}: {e}")
    
print(f"tile files: {len(filenames)}")
print(f"emoji tiles loaded: {len(ecabulary)}")
print(f"tile_size: {tile_size}")
print(f"patch_size: {patch_size}")

def best_emoji(img_patch):
  # rmse lower better, 0.0 identical
  measure_fn, comp_fn = metric_functions['rmse'], lambda candidate, prior : candidate < prior
  # fsim higher better, 1.0 identical - so slow no output?
  #measure_fn, comp_fn = metric_functions['fsim'], lambda candidate, prior : candidate > prior
  # psnr higher better, inf identical - seems competitive with rmse in speed
  # measure_fn, comp_fn = metric_functions['psnr'], lambda candidate, prior : candidate > prior
  # sre higher better, inf identical - very speedy, didn't capture colors/lights as well on mario
  #measure_fn, comp_fn = metric_functions['sre'], lambda candidate, prior : candidate > prior
  # ssim higher better, 1.0 identical - slower than rmse
  #measure_fn, comp_fn = metric_functions['ssim'], lambda candidate, prior : candidate > prior
  # uiq higher better, identical only got 0.94 ?! - so slow no output?
  #measure_fn, comp_fn = metric_functions['uiq'], lambda candidate, prior : candidate > prior

  best_yet = None
  best_score = None
  # TODO: offer/save=aside top-N matches with scores
  for k, (f, tile) in ecabulary.items():
    score = measure_fn(img_patch, tile)
    if best_yet == None or comp_fn(score , best_score):
      best_score = score
      best_yet = k
  return (best_yet, best_score)

img_filename = args.in_image
in_img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)  # TODO: support transparency, config bg color
print(f"{img_filename}: {in_img.shape}")
if args.bg_color: 
  in_img = solid_background(in_img, bgr_bgcolor)
  print(f"{img_filename} bg={rgb_bgcolor}: {in_img.shape}")
emoji_grid = (ceil(in_img.shape[0]/patch_size[0]), ceil(in_img.shape[1]/patch_size[1]))
grid_glyphs = emoji_grid[0]*emoji_grid[1]
print(f"emoji grid: {emoji_grid} ({grid_glyphs} glyphs, or {grid_glyphs+emoji_grid[1]} with newlines")

# TODO: use multiple threads
for x in range(0, in_img.shape[0], patch_size[0]):
  for y in range(0, in_img.shape[1], patch_size[1]):
    patch = in_img[x:x+patch_size[0], y:y+patch_size[1]]
    # TODO: notice identical patch, reuse last emoji
    patch = cv2.resize(patch, tile_size)
    e, score = best_emoji(patch)
    # print(f"{x,y}: {e} @ {score} {type(e)} {len(e)} {ecabulary[e][0]}")
    sys.stdout.write(e)
  print('')
