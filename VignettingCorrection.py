from __future__ import print_function

import argparse
import cv2
import numpy as np
import os

def get_filename_parts(fn):
    p = os.path.split(fn)

    f = os.path.splitext(p[1])

    return [ p[0], f[0], f[1] ]

class VignettingCorrector(object):
    def __init__(self, coeff):
        super(VignettingCorrector, self).__init__()

        self.coeff  = coeff.astype(np.float32)
        self.coeff3 = np.stack((coeff, coeff, coeff), axis=-1).astype(np.float32)
    
    def _correct(self, img, coeff, scale):
        coeff = 1.0 + (coeff - 1.0)*scale

        return np.clip( img.astype(np.float32) * coeff, 0.0, 255.0 ).astype(np.uint8)

    def correct(self, img, scale):
        if ( 2 == len(img.shape) ):
            return self._correct(img, self.coeff, scale)
        elif ( 3 == len(img.shape) ):
            return self._correct(img, self.coeff3, scale)
        else:
            raise Exception("Expecting len(img.shape) == %d. " % (len(img.shape)))

if __name__ == "__main__":
    print("Correct vignetting effect.")

    parser = argparse.ArgumentParser(description="Correct the vignetting effect of an image by using a coefficient mask.")

    parser.add_argument("inimage", type=str, \
        help="The input image.")
    parser.add_argument("mask", type=str, \
        help="The coefficient mask in .npy format.")
    parser.add_argument("--out-image", type=str, default="", \
        help="The corrected image. Leave blank for auto naming.")
    parser.add_argument("--scale", type=float, default=1.0, \
        help="The additional scale on the coefficients.")

    args = parser.parse_args()

    # Argument check.
    if ( args.scale < 0 ):
        raise Exception("args.scale == %f, it should be non-negative. " % (args.scale))

    # Open the input files.
    if ( not os.path.isfile( args.inimage ) ):
        raise Exception("%s does not exist. " % ( args.inimage ))

    imgOri = cv2.imread(args.inimage, cv2.IMREAD_UNCHANGED)
    print("%s read. " % (args.inimage))

    if ( not os.path.isfile( args.mask ) ):
        raise Exception("%s does not exist. " % ( args.mask ))

    coeff = np.load(args.mask).astype(np.float32)
    print("%s read. " % (args.mask))

    # The output file.
    if ( "" == args.out_image ):
        parts = get_filename_parts( args.inimage )
        outFn = "%s/%s_VC.png" % ( parts[0], parts[1] )
    else:
        outFn = args.out_image

    # Test the output directory.
    parts = get_filename_parts(outFn)

    if ( not os.path.isdir(parts[0]) ):
        os.makedirs(parts[0])
        print("Create %s. " % ( parts[0] ))
    
    # Vignetting correction.
    corrector = VignettingCorrector(coeff)

    imgNew = corrector.correct(imgOri, args.scale)

    # Save the image.
    cv2.imwrite( outFn, imgNew, [cv2.IMWRITE_PNG_COMPRESSION, 0] )

    print("%s saved. " % (outFn))

    print("Done.")
