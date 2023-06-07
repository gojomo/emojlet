from image_similarity_measures.quality_metrics import metric_functions

import cv2
import numpy as np

import os
import sys
import argparse

from imsim_tiler import ImSimTiler, Dims

parser = argparse.ArgumentParser(
  prog="emojlet",
  description="render image as emoji tiles",
)
parser.add_argument('tiles_dir', help="directory with emoji image files")
parser.add_argument('in_image_path', help="input image")
parser.add_argument('--tile', dest='tile', help="tile w[,h]", default='18')
parser.add_argument('--patch', dest='patch', help="patch w[,h]", default='64,78')
parser.add_argument('--bg', dest='bg_color',
                    help="fill transparent areas with RRGGBB value", default=None)

args = parser.parse_args()

def dims_from_spec(dimspec):
  """read two ints separate by comma; if one int, repeat it for square dims"""
  dims = tuple(int(dim) for dim in dimspec.split(','))
  if len(dims)==1:
    dims = (dims[0], dims[0])
  return Dims(w=dims[0], h=dims[1])

tile_size = dims_from_spec(args.tile)
patch_size = dims_from_spec(args.patch)
if args.bg_color:
  rgb_bgcolor = tuple(int(args.bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
  bgr_bgcolor = (rgb_bgcolor[2], rgb_bgcolor[1], rgb_bgcolor[0])
else:
  bgr_bgcolor=None

print(f"tile_size: {tile_size}")
print(f"patch_size: {patch_size}")

tiler = ImSimTiler(args.in_image_path, patch_size, args.tiles_dir, tile_size, bgr_bgcolor)

in_img = tiler.working_in_image
#cv2.imwrite("imtemp.png", in_img)

# TODO: use multiple threads
for y in range(0, in_img.shape[0], patch_size.h):
  for x in range(0, in_img.shape[1], patch_size.w):
    raw_patch = in_img[y:y+patch_size.h, x:x+patch_size.w]
    e, score = tiler.best_emoji(raw_patch)
    # print(f"{x,y}: {e} @ {score} {type(e)} {len(e)} {ecabulary[e][0]}")
    sys.stdout.write(e)
  print('')

tiler.end()