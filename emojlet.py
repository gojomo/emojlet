from image_similarity_measures.quality_metrics import metric_functions

import cv2

import os
import sys

if len(sys.argv) < 2:
    print(f"usage: python {sys.argv[0]} image_dir target_image")
    exit(1)

ecabulary = {}
filenames = os.listdir(sys.argv[1])
tile_size = (18, 18)  # TODO: make configurable?

for f in sorted(filenames):
    if not f.endswith(".png"):
      continue
    emoji_tile = cv2.imread(os.path.join(sys.argv[1], f)) #, cv2.IMREAD_UNCHANGED)  # TODO cv.IMREAD_UNCHANGED to save transparency
    if len(emoji_tile.shape) != 3:
        print(emoji_tile.shape)
        continue
    try:
      # TODO: handle files with extra underscores
      emoji_zwjseq = ''.join(chr(int(hexstr, 16)) for hexstr in f.split("_")[-1].split(".")[0].split("-"))
      ecabulary[emoji_zwjseq] = cv2.resize(emoji_tile, tile_size)
    except Exception as e:
      print(e)
      print(f)
    
print(f"tile files: {len(filenames}")
print(f"emoji tiles loaded: {len(ecabulary)}")

def best_emoji(img_patch):
  # rmse lower better, 0.0 identical
  measure_fn, best_score, comp_fn = metric_functions['rmse'], float('inf'), lambda candidate, prior : candidate < prior
  # fsim higher better, 1.0 identical - so slow no output?
  #measure_fn, best_score, comp_fn = metric_functions['fsim'], float('-inf'), lambda candidate, prior : candidate > prior
  # psnr higher better, inf identical - seems competitive with rmse in speed
  # measure_fn, best_score, comp_fn = metric_functions['psnr'], float('-inf'), lambda candidate, prior : candidate > prior
  # sre higher better, inf identical - very speedy, didn't capture colors/lights as well on mario
  #measure_fn, best_score, comp_fn = metric_functions['sre'], float('-inf'), lambda candidate, prior : candidate > prior
  # ssim higher better, 1.0 identical - slower than rmse
  #measure_fn, best_score, comp_fn = metric_functions['ssim'], float('-inf'), lambda candidate, prior : candidate > prior
  # uiq higher better, identical only got 0.94 ?! - so slow no output?
  #measure_fn, best_score, comp_fn = metric_functions['uiq'], -999.9, lambda candidate, prior : candidate > prior

  best_yet = '⬛️'
  # TODO: offer top-N matches with scores
  for k, tile in ecabulary.items():
    score = measure_fn(img_patch, tile)
    if comp_fn(score , best_score):
      best_score = score
      best_yet = k
  return best_yet

patch_size = (36, 36)  # TODO make configurable

if len(sys.argv) == 3:
    img_filename = sys.argv[2]
    in_img = cv2.imread(img_filename) #, cv2.IMREAD_UNCHANGED)  # TODO: support transparency, config bg color
    print(f"{img_filename}: {in_img.shape}")

    # TODO: use multiple threads
    for x in range(0, in_img.shape[0], patch_size[0]):
      for y in range(0, in_img.shape[1], patch_size[1]):
        patch = in_img[x:x+patch_size[0], y:y+patch_size[1]]
        # TODO: notice identical patch, reuse last emoji
        patch = cv2.resize(patch, tile_size)
        e = best_emoji(patch)
        sys.stdout.write(e)
      print('')
else:
    print("\nerror, needs image_dir & target_image")
