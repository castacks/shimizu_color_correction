# Use p3cv4.

import argparse
import cv2
import glob
import numpy as np
import os
from threading import Thread

class ThreadWhitebalance(Thread):
    def __init__(self, name, outDir, files, idxStart, idxEnd, bf, vcMask):
        super(ThreadWhitebalance, self).__init__()

        self.name     = name     # The name of this thread
        self.outDir   = outDir   # The output directory, must be present.
        self.files    = files    # The list of filenames.
        self.idxStart = idxStart # The starting index of self.files.
        self.idxEnd   = idxEnd   # The ending index of self.files, inclusive.
        self.bf       = bf       # The NumPy array of the balancing factors. BGR order.
        self.vcMask   = vcMask   # The mask for vignetting effect.

    def run(self):
        count = 1

        images  = self.files[self.idxStart:self.idxEnd+1]
        nImages = len(images)

        for img in images:
            print("%s: Process (%d/%d) %s." % ( self.name, count, nImages, img ))

            # Load the image.
            cvImg = cv2.imread( img, cv2.IMREAD_UNCHANGED )

            # Dummy image.
            dummy = np.zeros_like( cvImg[:, :, 0] )
            
            # Balance the input image
            for i in range( cvImg.shape[2] ):
                cvImg[:, :, i] = cv2.scaleAdd( cvImg[:, :, i], self.bf[i], dummy )
                cvImg[:, :, i] = cv2.multiply( cvImg[:, :, i], self.vcMask, dtype=cv2.CV_8UC1 )

            # Get the name components of the file name.
            fn  = os.path.split( img )[1]
            ext = os.path.splitext( fn )[1]
            fn  = os.path.splitext( fn )[0]

            # Save the balanced image.
            cv2.imwrite( self.outDir + "/" + fn + ext, cvImg )

            count += 1

if __name__ == "__main__":
    print("Batch white balance images.")

    parser = argparse.ArgumentParser(description='White balance all the images inside a folder.')
    parser.add_argument("--input-dir", type=str, help="The input directory.")
    parser.add_argument("--search-pattern", type=str, default="*", help="The the searching pattern.")
    parser.add_argument("--ext", type=str, default="bmp", help="The file extension to be found.")
    parser.add_argument("--output-dir", type=str, help="The output directory.", default="./")
    parser.add_argument("--bf", type=str, default="bf.dat", help="The file contains the balancing factors.")
    parser.add_argument("--skip-wb", action="store_true", default=False, help="Skip white balance.")
    parser.add_argument("--vc", action="store_true", default=False, help="Perform vignetting-correction.")
    parser.add_argument("--vc-mask", type=str, default="mask.npy", help="The vignetting-correction mask saved as a NumPy binary file. Only effective is --vc is set.")
    parser.add_argument("--np", type=int, default=2, help="Number of threads.")

    args = parser.parse_args()

    # Test if the output directory exists.
    if ( False == os.path.isdir( args.output_dir ) ):
        os.makedirs( args.output_dir )

    # Find all the images.
    images = glob.glob( args.input_dir + "/" + args.search_pattern + "." + args.ext )
    nImages = len(images)
    print("%d images are found at %s." % ( nImages, args.input_dir ))

    if ( 0 == nImages ):
        Exception("No images are found at %s with extension %s." % ( args.input_dir, args.ext ))

    if ( False == args.skip_wb ):
        # Load the balancing factors.
        bf = np.loadtxt( args.bf, dtype=np.float )
    else:
        # Dummy balancing factors.
        bf = np.ones( [3], dtype=np.float )
    
    print("The balancing factors are {}.".format( bf ))

    # Vignetting-correction.
    if ( True == args.vc ):
        # Load the mask.
        vcMask = np.load( args.vc_mask )
        print("Vignetting-correction mask loaded from %s." % ( args.vc_mask ))
    else:
        # Make a dummy mask.
        img0 = cv2.imread( images[0], cv2.IMREAD_UNCHANGED )
        vcMask = np.ones( [ img0.shape[0], img0.shape[1] ], dtype=np.float )

    # Figure out the index for each thread.
    if ( args.np <= 0 ):
        raise Exception("args.np = %d." % (args.np))
    
    idxStep = int(nImages / args.np)

    start = 0
    end   = idxStep-1
    idxStart = []
    idxEnd   = []
    for i in range(0, args.np-1):
        idxStart.append(start)
        idxEnd.append(end)
        start += idxStep
        end   += idxStep
    
    if ( idxEnd[-1] >= nImages ):
        idxEnd[-1] = nImages - 1
    else:
        idxStart.append(start)
        idxEnd.append(nImages-1)

    # Create the threads.
    threads = []
    for i in range(args.np):
        threads.append( \
            ThreadWhitebalance( "T%02d" % (i), \
                args.output_dir, images, idxStart[i], idxEnd[i], bf, vcMask ) )

    # Start all the threads.
    for t in threads:
        t.start()
    
    print("All threads started.")

    # Join all the threads.
    for t in threads:
        t.join()

    print("All threads joined.")

    # # White balance all the found images.
    # count = 1
    # for img in images:
    #     print("Process (%d/%d) %s." % ( count, nImages, img ))

    #     # Load the image.
    #     cvImg = cv2.imread( img, cv2.IMREAD_UNCHANGED )
        
    #     # Balance the input image
    #     for i in range( cvImg.shape[2] ):
    #         cvImg[:, :, i] = cv2.scaleAdd( cvImg[:, :, i], bf[i], np.zeros_like( cvImg[:, :, i] ) )
    #         cvImg[:, :, i] = cv2.multiply( cvImg[:, :, i], vcMask, dtype=cv2.CV_8UC1 )

    #     # Get the name components of the file name.
    #     fn  = os.path.split( img )[1]
    #     ext = os.path.splitext( fn )[1]
    #     fn  = os.path.splitext( fn )[0]

    #     # Save the balanced image.
    #     # cv2.imwrite( args.output_dir + "/" + fn + "_Balanced" + ext, cvImg )
    #     cv2.imwrite( args.output_dir + "/" + fn + ext, cvImg )

    #     count += 1
