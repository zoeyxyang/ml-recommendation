
import os
import argparse
import skimage as ski
import numpy as np


def pad_image(image, pad_height, pad_width):
  if pad_width > image.shape[1]:
    width = pad_width
    w_i = (pad_width//2) - image.shape[1]//2
    w_f = w_i + image.shape[1]
  else:
    width = image.shape[1]
    w_i = 0
    w_f = image.shape[1]
  if pad_height > image.shape[0]:
    height = pad_height
    h_i = (pad_height//2) - image.shape[0]//2
    h_f = h_i + image.shape[0]
  else:
    height = image.shape[0]
    h_i = 0
    h_f = image.shape[0]
  padded_image = np.zeros((height,width,3)).astype(image.dtype)
  #padded_image[:] = 250
  padded_image[h_i:h_f, w_i:w_f] = image
  return padded_image


if __name__ == "__main__":

  parser = argparse.ArgumentParser("preprocess images")
  parser.add_argument("-i","--input_list", required=True, type=str, help="text file containing list of original images")
  parser.add_argument("-n","--n_images", type=int, default=-1, help="number of images to process (do all if negative)")
  parser.add_argument("-p","--print_freq", type=int, default=1000, help="print progress after this many entries")
  parser.add_argument("-wP","--padded_width", type=int, default=64, help="final image pixel width, zero-pad to this if original image is smaller. Set to negative value to skip zero padding")
  parser.add_argument("-hP","--padded_height", type=int, default=64, help="final image pixel height, zero-pad to this if original image is smaller. Set to negative value to skip zero padding")
  parser.add_argument("-wR","--resize_width", type=int, default=43, help="pre-pad image pixel width, scale to this before zero padding, don't scale if negative")
  parser.add_argument("-hR","--resize_height", type=int, default=64, help="pre-pad image pixel height, scale to this before zero padding, don't scale if negative")
  args = parser.parse_args()
  #117, 175 --> 43 ,64 --> 64, 64

  with open(args.input_list,"r") as image_files:
    for i, image_file in enumerate(image_files):
      if i >= args.n_images and args.n_images > 0:
        break
      if i % args.print_freq == 0:
        print(f"reached image {i}")
      image = ski.io.imread(image_file.strip())
      #scale: resize to a smaller size and also convert it into an float-image(pixel value between 0 and 1)
      if args.resize_height > 0 or args.resize_width > 0:
        imgH = args.resize_height if (args.resize_height > 0) else image.shape[0]
        imgW = args.resize_width if (args.resize_width > 0) else image.shape[1]
        image = ski.transform.resize(image, (imgH,imgW))
      #zero pad
      if args.padded_width > 0 or args.padded_height > 0:
        image = pad_image(image, args.padded_height, args.padded_width)
      #change image format to 0->1 floating point vals ## divide by 255
      #image = ski.util.img_as_float(image)/255
      #image = ski.util.img_as_float(image)
      #permute dimensions to get (channel, height, width)
      image = np.transpose(image, (2,0,1))
      #save to binary .npy file
      image_dir = os.path.dirname(image_file.strip())
      image_name = os.path.basename(image_file.strip())
      if "/images/" not in image_dir:
        sys.exit("\"images\" directory required in path to input images")
      output_dir = image_dir.replace("/images/","/preprocessed_images/")
      output_name = image_name.replace(".jpg",".npy")
      os.system(f"mkdir -p {output_dir}")
      np.save(f"{output_dir}/{output_name}", image)


