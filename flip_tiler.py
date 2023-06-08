# policy for mapping image patches to emoji characters based on 
# image-similarity-metrics

from image_similarity_measures.quality_metrics import metric_functions

import cv2
cv = cv2
from PIL import Image
import numpy as np

import os

from math import ceil
from collections import namedtuple

Dims = namedtuple('Dims', ['w', 'h'])  # width, height

Emojus = namedtuple('Emojus', [
                                'cseq',     # characters
                                'basename', # core name
                                'fullname', # name with modifiers like skin tones
                                'filename', # source filename
                                'tile',     # image representation
                              ])  

SpatialFilters = namedtuple("SpatialFilters", ['s_a', 's_rg', 's_by'])
filters_by_ppd = {}

PreprocessedImage = namedtuple("PreprocessedImage",
                               [
                                'perceptually_uniform',
                                'edges',
                                'points',
                               ])

########

def color_space_transform(input_color, fromSpace2toSpace):
	"""
	Transforms inputs between different color spaces

	:param input_color: tensor of colors to transform (with CxHxW layout)
	:param fromSpace2toSpace: string describing transform
	:return: transformed tensor (with CxHxW layout)
	"""
	dim = input_color.shape

	# Assume D65 standard illuminant
	reference_illuminant = np.array([[[0.950428545]], [[1.000000000]], [[1.088900371]]]).astype(np.float32)
	inv_reference_illuminant = np.array([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).astype(np.float32)

	if fromSpace2toSpace == "srgb2linrgb":
		limit = 0.04045
		transformed_color = np.where(input_color > limit, np.power((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

	elif fromSpace2toSpace == "linrgb2srgb":
		limit = 0.0031308
		transformed_color = np.where(input_color > limit, 1.055 * (input_color ** (1.0 / 2.4)) - 0.055, 12.92 * input_color)

	elif fromSpace2toSpace == "linrgb2xyz" or fromSpace2toSpace == "xyz2linrgb":
		# Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
		# Assumes D65 standard illuminant
		if fromSpace2toSpace == "linrgb2xyz":
			a11 = 10135552 / 24577794
			a12 = 8788810  / 24577794
			a13 = 4435075  / 24577794
			a21 = 2613072  / 12288897
			a22 = 8788810  / 12288897
			a23 = 887015   / 12288897
			a31 = 1425312  / 73733382
			a32 = 8788810  / 73733382
			a33 = 70074185 / 73733382
		else:
			# Constants found by taking the inverse of the matrix
			# defined by the constants for linrgb2xyz
			a11 = 3.241003275
			a12 = -1.537398934
			a13 = -0.498615861
			a21 = -0.969224334
			a22 = 1.875930071
			a23 = 0.041554224
			a31 = 0.055639423
			a32 = -0.204011202
			a33 = 1.057148933
		A = np.array([[a11, a12, a13],
					  [a21, a22, a23],
					  [a31, a32, a33]]).astype(np.float32)

		input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
		transformed_color = np.matmul(A, input_color)
		transformed_color = np.transpose(transformed_color, (1, 2, 0))

	elif fromSpace2toSpace == "xyz2ycxcz":
		input_color = np.multiply(input_color, inv_reference_illuminant)
		y = 116 * input_color[1:2, :, :] - 16
		cx = 500 * (input_color[0:1, :, :] - input_color[1:2, :, :])
		cz = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])
		transformed_color = np.concatenate((y, cx, cz), 0)

	elif fromSpace2toSpace == "ycxcz2xyz":
		y = (input_color[0:1, :, :] + 16) / 116
		cx = input_color[1:2, :, :] / 500
		cz = input_color[2:3, :, :] / 200

		x = y + cx
		z = y - cz
		transformed_color = np.concatenate((x, y, z), 0)

		transformed_color = np.multiply(transformed_color, reference_illuminant)

	elif fromSpace2toSpace == "xyz2lab":
		input_color = np.multiply(input_color, inv_reference_illuminant)
		delta = 6 / 29
		delta_square = delta * delta
		delta_cube = delta * delta_square
		factor = 1 / (3 * delta_square)

		input_color = np.where(input_color > delta_cube, np.power(input_color, 1 / 3), (factor * input_color + 4 / 29))

		l = 116 * input_color[1:2, :, :] - 16
		a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
		b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

		transformed_color = np.concatenate((l, a, b), 0)

	elif fromSpace2toSpace == "lab2xyz":
		y = (input_color[0:1, :, :] + 16) / 116
		a =  input_color[1:2, :, :] / 500
		b =  input_color[2:3, :, :] / 200

		x = y + a
		z = y - b

		xyz = np.concatenate((x, y, z), 0)
		delta = 6 / 29
		factor = 3 * delta * delta
		xyz = np.where(xyz > delta,  xyz ** 3, factor * (xyz - 4 / 29))

		transformed_color = np.multiply(xyz, reference_illuminant)

	elif fromSpace2toSpace == "srgb2xyz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color,'linrgb2xyz')
	elif fromSpace2toSpace == "srgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "linrgb2ycxcz":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
	elif fromSpace2toSpace == "srgb2lab":
		transformed_color = color_space_transform(input_color, 'srgb2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "linrgb2lab":
		transformed_color = color_space_transform(input_color, 'linrgb2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	elif fromSpace2toSpace == "ycxcz2linrgb":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
	elif fromSpace2toSpace == "lab2srgb":
		transformed_color = color_space_transform(input_color, 'lab2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
		transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
	elif fromSpace2toSpace == "ycxcz2lab":
		transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
		transformed_color = color_space_transform(transformed_color, 'xyz2lab')
	else:
		sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)

	return transformed_color

##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################
# LDR-FLIP functions
##################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################

def generate_spatial_filter(pixels_per_degree, channel):
	"""
	Generates spatial contrast sensitivity filters with width depending on
	the number of pixels per degree of visual angle of the observer

	:param pixels_per_degree: float indicating number of pixels per degree of visual angle
	:param channel: string describing what filter should be generated
	:yield: Filter kernel corresponding to the spatial contrast sensitivity function of the given channel
	"""
	a1_A = 1
	b1_A = 0.0047
	a2_A = 0
	b2_A = 1e-5 # avoid division by 0
	a1_rg = 1
	b1_rg = 0.0053
	a2_rg = 0
	b2_rg = 1e-5 # avoid division by 0
	a1_by = 34.1
	b1_by = 0.04
	a2_by = 13.5
	b2_by = 0.025
	if channel == "A": # Achromatic CSF
		a1 = a1_A
		b1 = b1_A
		a2 = a2_A
		b2 = b2_A
	elif channel == "RG": # Red-Green CSF
		a1 = a1_rg
		b1 = b1_rg
		a2 = a2_rg
		b2 = b2_rg
	elif channel == "BY": # Blue-Yellow CSF
		a1 = a1_by
		b1 = b1_by
		a2 = a2_by
		b2 = b2_by

	# Determine evaluation domain
	max_scale_parameter = max([b1_A, b2_A, b1_rg, b2_rg, b1_by, b2_by])
	r = np.ceil(3 * np.sqrt(max_scale_parameter / (2 * np.pi**2)) * pixels_per_degree)
	r = int(r)
	deltaX = 1.0 / pixels_per_degree
	x, y = np.meshgrid(range(-r, r + 1), range(-r, r + 1))
	z = ((x * deltaX)**2 + (y * deltaX)**2).astype(np.float32)

	# Generate weights
	s = a1 * np.sqrt(np.pi / b1) * np.exp(-np.pi**2 * z / b1) + a2 * np.sqrt(np.pi / b2) * np.exp(-np.pi**2 * z / b2)
	s = s / np.sum(s)

	return s

def spatial_filter(img, s_a, s_rg, s_by):
	"""
	Filters an image with channel specific spatial contrast sensitivity functions
	and clips result to the unit cube in linear RGB

	:param img: image to filter (with CxHxW layout in the YCxCz color space)
	:param s_a: spatial filter matrix for the achromatic channel
	:param s_rg: spatial filter matrix for the red-green channel
	:param s_by: spatial filter matrix for the blue-yellow channel
	:return: input image (with CxHxW layout) transformed to linear RGB after filtering with spatial contrast sensitivity functions
	"""
	# Apply Gaussian filters
	dim = img.shape
	img_tilde_opponent = np.zeros((dim[0], dim[1], dim[2])).astype(np.float32)
	img_tilde_opponent[0:1, :, :] = cv.filter2D(img[0:1, :, :].squeeze(0), ddepth=-1, kernel=s_a, borderType=cv.BORDER_REPLICATE)
	img_tilde_opponent[1:2, :, :] = cv.filter2D(img[1:2, :, :].squeeze(0), ddepth=-1, kernel=s_rg, borderType=cv.BORDER_REPLICATE)
	img_tilde_opponent[2:3, :, :] = cv.filter2D(img[2:3, :, :].squeeze(0), ddepth=-1, kernel=s_by, borderType=cv.BORDER_REPLICATE)

	# Transform to linear RGB for clamp
	img_tilde_linear_rgb = color_space_transform(img_tilde_opponent, 'ycxcz2linrgb')

	# Clamp to RGB box
	return np.clip(img_tilde_linear_rgb, 0.0, 1.0)

def hunt_adjustment(img):
	"""
	Applies Hunt-adjustment to an image

	:param img: image to adjust (with CxHxW layout in the L*a*b* color space)
	:return: Hunt-adjusted image (with CxHxW layout in the Hunt-adjusted L*A*B* color space)
	"""
	# Extract luminance component
	L = img[0:1, :, :]

	# Apply Hunt adjustment
	img_h = np.zeros(img.shape).astype(np.float32)
	img_h[0:1, :, :] = L
	img_h[1:2, :, :] = np.multiply((0.01 * L), img[1:2, :, :])
	img_h[2:3, :, :] = np.multiply((0.01 * L), img[2:3, :, :])

	return img_h

def hyab(reference, test):
	"""
	Computes the HyAB distance between reference and test images

	:param reference: reference image (with CxHxW layout in the standard or Hunt-adjusted L*A*B* color space)
	:param test: test image (with CxHxW layout in the standard or Hunt-adjusted L*A*B* color space)
	:return: matrix (with 1xHxW layout) containing the per-pixel HyAB distance between reference and test
	"""
	delta = reference - test
	return abs(delta[0:1, :, :]) + np.linalg.norm(delta[1:3, :, :], axis=0)

def redistribute_errors(power_deltaE_hyab, cmax):
	"""
	Redistributes exponentiated HyAB errors to the [0,1] range

	:param power_deltaE_hyab: float containing the exponentiated HyAb distance
	:param cmax: float containing the exponentiated, maximum HyAB difference between two colors in Hunt-adjusted L*A*B* space
	:return: matrix (on 1xHxW layout) containing redistributed per-pixel HyAB distances (in range [0,1])
	"""
	# Set redistribution parameters
	pc = 0.4
	pt = 0.95

	# Re-map error to 0-1 range. Values between 0 and
	# pccmax are mapped to the range [0, pt],
	# while the rest are mapped to the range (pt, 1]
	deltaE_c = np.zeros(power_deltaE_hyab.shape)
	pccmax = pc * cmax
	deltaE_c = np.where(power_deltaE_hyab < pccmax, (pt / pccmax) * power_deltaE_hyab, pt + ((power_deltaE_hyab - pccmax) / (cmax - pccmax)) * (1.0 - pt))

	return deltaE_c

def feature_detection(imgy, pixels_per_degree, feature_type):
	"""
	Detects edges and points (features) in the achromatic image

	:param imgy: achromatic image (on 1xHxW layout, containing normalized Y-values from YCxCz)
	:param pixels_per_degree: float describing the number of pixels per degree of visual angle of the observer
	:param feature_type: string indicating the type of feature to detect
	:return: tensor (with layout 2xHxW with values in range [0,1]) containing large values where features were detected
	"""
	# Set peak to trough value (2x standard deviations) of human edge
	# detection filter
	w = 0.082

	# Compute filter radius
	sd = 0.5 * w * pixels_per_degree
	radius = int(np.ceil(3 * sd))

	# Compute 2D Gaussian
	[x, y] = np.meshgrid(range(-radius, radius+1), range(-radius, radius+1))
	g = np.exp(-(x ** 2 + y ** 2) / (2 * sd * sd))

	if feature_type == 'edge': # Edge detector
		# Compute partial derivative in x-direction
		Gx = np.multiply(-x, g)
	else: # Point detector
		# Compute second partial derivative in x-direction
		Gx = np.multiply(x ** 2 / (sd * sd) - 1, g)

	# Normalize positive weights to sum to 1 and negative weights to sum to -1
	negative_weights_sum = -np.sum(Gx[Gx < 0])
	positive_weights_sum = np.sum(Gx[Gx > 0])
	Gx = np.where(Gx < 0, Gx / negative_weights_sum, Gx / positive_weights_sum)

	# Detect features
	featuresX = cv.filter2D(imgy.squeeze(0), ddepth=-1, kernel=Gx, borderType=cv.BORDER_REPLICATE)
	featuresY = cv.filter2D(imgy.squeeze(0), ddepth=-1, kernel=np.transpose(Gx), borderType=cv.BORDER_REPLICATE)

	return np.stack((featuresX, featuresY))

########


def ldrflip_components(img, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180):
  """return all aspects of img separately pre-calculable"""

  cst_img = color_space_transform(img, 'srgb2ycxcz')
  if pixels_per_degree not in filters_by_ppd:
    s_a = generate_spatial_filter(pixels_per_degree, 'A')
    s_rg = generate_spatial_filter(pixels_per_degree, 'RG')
    s_by = generate_spatial_filter(pixels_per_degree, 'BY')
    filters_by_ppd[pixels_per_degree] = SpatialFilters(s_a=s_a, s_rg=s_rg, s_by=s_by)

  precalced_filter = filters_by_ppd[pixels_per_degree]
  filtered_img = spatial_filter(cst_img, precalced_filter.s_a, precalced_filter.s_rg, precalced_filter.s_by)
  preprocessed_img = hunt_adjustment(color_space_transform(filtered_img, 'linrgb2lab'))

  achromatic_img = (cst_img[0:1, :, :] + 16) / 116
  edges_img = feature_detection(achromatic_img, pixels_per_degree, 'edge')
  points_img = feature_detection(achromatic_img, pixels_per_degree, 'point')

  return PreprocessedImage(perceptually_uniform=preprocessed_img,
                           edges=edges_img,
                           points=points_img)

# precalcs that don't depend on images
hunt_adjusted_green = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[1.0]], [[0.0]]]).astype(np.float32), 'linrgb2lab'))
hunt_adjusted_blue = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[0.0]], [[1.0]]]).astype(np.float32), 'linrgb2lab'))

def compute_ldrflip(ref, test, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180):
  ref_prep = ldrflip_components(ref)
  test_prep = ldrflip_components(test)

  return compute_ldrflip_prepped(ref_prep, test_prep)

def compute_ldrflip_prepped(ref_prep, test_prep):
  preprocessed_reference = ref_prep.perceptually_uniform
  preprocessed_test = test_prep.perceptually_uniform
  # Set color and feature exponents
  qc = 0.7
  qf = 0.5
  # Color metric
  deltaE_hyab = hyab(preprocessed_reference, preprocessed_test)
  cmax = np.power(hyab(hunt_adjusted_green, hunt_adjusted_blue), qc)
  deltaE_c = redistribute_errors(np.power(deltaE_hyab, qc), cmax)

  edges_reference, points_reference = ref_prep.edges, ref_prep.points
  edges_test, points_test = test_prep.edges, test_prep.points
  # Feature metric
  deltaE_f = np.maximum(abs(np.linalg.norm(edges_reference, axis=0) - np.linalg.norm(edges_test, axis=0)),
						  abs(np.linalg.norm(points_test, axis=0) - np.linalg.norm(points_reference, axis=0)))
  deltaE_f = np.power(((1 / np.sqrt(2)) * deltaE_f), qf)

  # --- Final error ---
  return np.power(deltaE_c, 1 - deltaE_f)

def compute_ldrflip_mean_prepped(ref_prep, test_prep):
  return np.mean(compute_ldrflip_prepped(ref_prep, test_prep))

def imequal(im1, im2):
  """return if 2 cv2-style images identical"""
  # TODO: exit earlier for common case (1st inequality rather than tallied norm)
  return im1.shape == im2.shape and not cv2.norm(im1, im2, cv2.NORM_L1)

def HWCtoCHW(x):
	"""
	Transforms an image from HxWxC layout to CxHxW

	:param x: image with HxWxC layout
	:return: image with CxHxW layout
	"""
	return np.rollaxis(x, 2)

def flip_format(img):
  """get into flip's CHW, float format"""
  img = np.asarray(img).astype(np.float32)
  img = HWCtoCHW(img)
  img = img / 255.0
  return img

class FlipTiler: 
  def __init__(self, in_image_path, patch_size, tiles_dir, tile_size, precedence, background_color=(0,0,0)):
    self.background_color = background_color
    self.in_image_path = in_image_path
    self.patch_size = patch_size
    self.tile_size = tile_size
    self.prep_image()

    self.tiles_dir = tiles_dir
    self.precedence = precedence
    self.read_tiles()

    # simple repeated-patch caching
    self.last_patch = None
    self.last_best = None
    self.reuses = 0
    self.ties = 0
    self.max_way_tie = 0

  def prep_image(self):
    """do any pre-analysis prep of image (eg fill transparent pixels wtih bg)"""

    self.raw_in_image = Image.open(self.in_image_path, 'r').convert('RGBA')
    bg_img = Image.new('RGBA', self.raw_in_image.size, self.background_color)
    self.working_in_image = Image.alpha_composite(bg_img, self.raw_in_image)  # fill transparent

    print(f"{self.in_image_path}: {self.working_in_image.size}")

    self.emoji_grid = Dims(w=ceil(self.raw_in_image.size[0]/self.patch_size.w), 
                      h=ceil(self.raw_in_image.size[1]/self.patch_size.h))
 
    # scale image to exactly fit tiling dimensions
    self.working_in_image = self.working_in_image.resize((self.emoji_grid.w*self.tile_size.w, 
                                                        self.emoji_grid.h*self.tile_size.h))
    print(f"{self.emoji_grid} grid tiling-ready: {self.working_in_image.size}")
    grid_glyph_count = self.emoji_grid[0]*self.emoji_grid[1]
    print(f"({grid_glyph_count} emojis, or {grid_glyph_count+self.emoji_grid[1]} glyphs with newlines")

    self.working_in_image = flip_format(self.working_in_image.convert('RGB'))

  def read_tiles(self):
    """read every PNG in given directory into a usable-for-matching in-memory image tile

       ecabulary is list of Emojus tuples
    """
    self.ecabulary = []

    filenames = os.listdir(self.tiles_dir)
    for f in filenames:
        if not f.endswith(".png"):
          continue
        #try:
        if True:
          fname, ext = f.split('.')
          undersplits = fname.split('_')
          if len(undersplits) == 2:
            emoji_chars = undersplits[-1]
            basename = fullname = undersplits[0]
          elif len(undersplits) == 4:
            emoji_chars = undersplits[-2]
            basename = undersplits[0]
            fullname = '_'.join(undersplits[0:2])
          else:
            print(f"ERROR: unparseable filename {f}; skipping")
            continue
          emoji_zwjseq = ''.join(chr(int(hexstr, 16)) for hexstr in emoji_chars.split("-"))
          if emoji_zwjseq == u'\uFE0F':
            # should not render, so unneeded for us
            continue
          #if emoji_zwjseq in self.ecabulary:
          #  print(f"DUPLICATE: {emoji_zwjseq} {f}")
          #  continue
          emoji_tile = Image.open(os.path.join(self.tiles_dir, f), 'r').convert('RGBA')
          bg_img = Image.new('RGBA', emoji_tile.size, self.background_color)
          emoji_tile = Image.alpha_composite(bg_img, emoji_tile)  # fill transparent
          emoji_tile = emoji_tile.resize(self.tile_size).convert('RGB')

          self.ecabulary.append(Emojus(cseq=emoji_zwjseq, 
                                       basename=basename, 
                                       fullname=fullname, 
                                       filename=f, 
                                       tile=ldrflip_components(flip_format(emoji_tile))))
        #except Exception as e:
        #  print(f"{f}: {e}")
    
    # sort, affecting which earlier tiles win ties
    if self.precedence == 'alpha':
      # filename
      self.ecabulary.sort(key=lambda emojus: emojus.filename)
    elif self.precedence == 'namelen':
      # shorter basename, shorter fullname, filename
      self.ecabulary.sort(key=lambda emojus: (len(emojus.basename), len(emojus.fullname), emojus.filename ))
    elif self.precedence == 'chars':
      self.ecabulary.sort(key=lambda emojus: emojus.cseq)
    elif self.precedence == 'age':
      # prefer older characters, then lower chars
      from unicode_age import unicode_age_data
      self.ecabulary.sort(key=lambda emojus: (unicode_age_data.get(emojus.cseq[0]), float('inf'), emojus.cseq))
    else: 
      # treat as int seed for a shuffle
      import random
      random.Random(int(self.precedence)).shuffle(self.ecabulary)

    print(f"tile files: {len(filenames)}")
    print(f"emoji tiles loaded: {len(self.ecabulary)} & ordered by {self.precedence}")

  

  def best_emoji(self, tilesized_patch):
    """ return (emoji, score) that best matches the supplied patch"""

    if self.last_patch is not None and imequal(self.last_patch, tilesized_patch):
      self.reuses += 1  # retain prior e, score
      return self.last_best

    best_yet = None
    best_score = None
    ties_for_best = 1
    tilesized_prep = ldrflip_components(tilesized_patch)
    # TODO: offer/save-aside top-N matches with scores for later fuzzing/subsetting
    for emojus in self.ecabulary:
      score = compute_ldrflip_mean_prepped(tilesized_prep, emojus.tile)
      if score == best_score:
        ties_for_best += 1
      if best_yet == None or score < best_score:
        ties_for_best = 1  # reset for new best
        best_score = score
        best_yet = emojus.cseq
    
    # cache & return
    self.last_patch = tilesized_patch
    self.last_best = (best_yet, best_score)
    if ties_for_best:
      self.ties += 1
      self.max_way_tie = max(self.max_way_tie, ties_for_best)
    return self.last_best

  def end(self):
    """final reporting/cleanup"""
    print(f"reuses: {self.reuses}")
    print(f"ties: {self.ties}")
    print(f"max_way_tie: {self.max_way_tie}")

