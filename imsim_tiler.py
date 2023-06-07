# policy for mapping image patches to emoji characters based on 
# image-similarity-metrics

from image_similarity_measures.quality_metrics import metric_functions

import cv2
import numpy as np

import os

from math import ceil
from collections import namedtuple

Dims = namedtuple('Dims', ['w', 'h'])

def solid_background(foreground, bgcolor):
  """fill any transparent areas of 'foreground' with solid color bgcolor

     cv2-style images of (h, w, c) - & usually with BGRA colors
  """

  background = np.full(foreground.shape[:2] + (3,), bgcolor, np.uint8)

  # separate fg alpha & color channels
  foreground_colors = foreground[:, :, :3]
  alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0
  alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

  # combine background with alpha-weighted overlay
  return background * (1 - alpha_mask) + foreground_colors * alpha_mask


def imequal(im1, im2):
  """return if 2 cv2-style images identical"""
  # TODO: exit earlier for common case (1st inequality rather than tallied norm)
  return im1.shape == im2.shape and not cv2.norm(im1, im2, cv2.NORM_L1)


class ImSimTiler: 
  def __init__(self, in_image_path, patch_size, tiles_dir, tile_size, background_color=None):
    self.background_color = background_color
    self.in_image_path = in_image_path
    self.patch_size = patch_size
    self.prep_image()

    self.tiles_dir = tiles_dir
    self.tile_size = tile_size
    self.read_tiles()

    # simple repeated-patch caching
    self.last_patch = None
    self.last_best = None
    self.reuses = 0
    self.ties = 0

    # similarity measures
    # rmse lower better, 0.0 identical
    self.measure_fn, self.comp_fn = metric_functions['rmse'], lambda candidate, prior : candidate < prior
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

  def prep_image(self):
    """do any pre-analysis prep of image (eg fill transparent pixels wtih bg)"""

    self.raw_in_image = self.working_in_image = cv2.imread(self.in_image_path, cv2.IMREAD_UNCHANGED)
    print(f"{self.in_image_path}: {self.working_in_image.shape}")

    if self.background_color and self.raw_in_image.shape[2] == 4:
      # fill background then drop transparency channel
      self.working_in_image = solid_background(self.working_in_image, self.background_color)[:,:,0:3]
      print(f"background-filled {self.background_color}: {self.working_in_image.shape}")

    emoji_grid = Dims(w=ceil(self.raw_in_image.shape[1]/self.patch_size.w), 
                      h=ceil(self.raw_in_image.shape[0]/self.patch_size.h))
 
    # scale image to exactly fit tiling dimensions
    # TODO: consider alternate approaches, like transparent or edge-extended padding
    self.working_in_image = cv2.resize(self.working_in_image, (emoji_grid.w*self.patch_size.w, emoji_grid.h*self.patch_size.h))
    print(f"{emoji_grid} grid tiling-ready: {self.working_in_image.shape}")
    grid_glyph_count = emoji_grid[0]*emoji_grid[1]
    print(f"({grid_glyph_count} emojis, or {grid_glyph_count+emoji_grid[1]} glyphs with newlines")


  def read_tiles(self):
    """read every PNG in given directory into a usable-for-matching in-memory image tile

       ecabulary is dict mapping (emoji-unicode-sequence -> (filename, cv2-style image))
    """
    self.ecabulary = {}

    filenames = os.listdir(self.tiles_dir)
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
          if emoji_zwjseq in self.ecabulary:
            print(f"DUPLICATE: {emoji_zwjseq} {f}")
            continue
          emoji_tile = cv2.imread(os.path.join(self.tiles_dir, f), cv2.IMREAD_UNCHANGED)  # TODO cv.IMREAD_UNCHANGED to save transparency
          if self.background_color:
            emoji_tile = solid_background(emoji_tile, self.background_color)
          if self.working_in_image.shape[2] == 3 and emoji_tile.shape[2] != 3:
            # trim alpha if source image doesn't have it
            emoji_tile = emoji_tile[:,:,0:3]
          self.ecabulary[emoji_zwjseq] = (f, cv2.resize(emoji_tile, self.tile_size))
        except Exception as e:
          print(f"{f}: {e}")
    
    print(f"tile files: {len(filenames)}")
    print(f"emoji tiles loaded: {len(self.ecabulary)}")

  

  def best_emoji(self, img_patch):
    """ return (emoji, score) that best matches the supplied patch"""

    if self.last_patch is not None and imequal(self.last_patch, img_patch):
      self.reuses += 1  # retain prior e, score
      return self.last_best

    tilesized_patch = cv2.resize(img_patch, self.tile_size) 
    best_yet = None
    best_score = None
    ties_for_best = 0
    # TODO: offer/save-aside top-N matches with scores for later fuzzing/subsetting
    for k, (f, tile) in self.ecabulary.items():
      score = self.measure_fn(tilesized_patch, tile)
      if score == best_score:
        ties_for_best += 1
      if best_yet == None or self.comp_fn(score , best_score):
        ties_for_best = 0  # reset for new best
        best_score = score
        best_yet = k
    
    # cache & return
    self.last_patch = img_patch
    self.last_best = (best_yet, best_score)
    if ties_for_best:
      self.ties += 1
    return self.last_best

  def end(self):
    """final reporting/cleanup"""
    print(f"reuses: {self.reuses}")
    print(f"ties: {self.ties}")

