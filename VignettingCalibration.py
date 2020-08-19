
# Use p3cv4.

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def calibrate_vignetting_mask(img, w=9):
    """
    This funcion takes a gray scale image and try to find a 6-order radial response function
    I = I_0 * ( a6 * r^6 + a4 * r^4 + a2 * r^2 + a0 ).

    Then return the mask as an floating point single channel image. The a6 - a0 parameters are
    returned as a NumPy array.
    """

    # Check if img is a single channel image.
    if ( 2 != len( img.shape ) ):
        Exception("calibrate_vignetting_mask() only accepts single channel image.")

    # Find the center index of the image.
    centerIdx = np.array( [ int(img.shape[0]/2), int(img.shape[1]/2) ], dtype=np.int)

    # Average around the center index.
    halfW = int(w/2)
    centerWindow = img[ centerIdx[0]-halfW : centerIdx[0] + halfW + 1, \
                        centerIdx[1]-halfW : centerIdx[1] + halfW + 1 ]

    centerIntensity = centerWindow.mean()

    # The intensity ratio matrix.
    ir = centerIntensity / img
    ir = ir.reshape([-1, 1])

    # Create two index vectors.
    idxX = np.linspace( 0, img.shape[1], num=img.shape[1], endpoint=False, dtype=np.float )
    idxY = np.linspace( 0, img.shape[0], num=img.shape[0], endpoint=False, dtype=np.float )
    
    idxX = idxX - centerIdx[1]
    idxY = idxY - centerIdx[0]

    # Coordinate matrices.
    [ mx, my ] = np.meshgrid( idxX, idxY)

    # Distance.
    r = np.sqrt( mx**2 + my**2 )

    # Reshape r.
    r = r.reshape([-1,])

    # Normalize.
    r = r / r.max()

    # Make the R matrix.
    R = np.stack([ r**6, r**4, r**2, np.ones_like(r, dtype=r.dtype) ], axis=1)
    RTR = R.transpose().dot( R )

    # Solve the least-square problem.
    # a = np.linalg.solve( RTR, R.transpose().dot( ir ) )

    a, residuals, rank, s = np.linalg.lstsq(R, ir)

    # Make the mask.
    M = R.dot( a )
    M = M.reshape( img.shape )

    return M, a

def plot_vignetting_correction_curve(a, n=2000, fn="VignettingCorrectionFactorCurve.png"):
    # x-axis.
    x = np.linspace( -1, 1, n )
    y = a[0] * x**6 + a[1] * x**4 + a[2] * x**2 + a[3]

    plt.plot( x, y )
    plt.xlabel("Dimensionless distance from image center")
    plt.ylabel("VC factor")
    plt.title("Vignetting-correction factor")
    plt.savefig(fn)
    plt.show()

def plot_vignetting_correction_coefficient_and_image(fn, img, a, n=2000):
    """
    
    """

    w = 9

    # Find the center index of the image.
    centerIdx = np.array( [ int(img.shape[0]/2), int(img.shape[1]/2) ], dtype=np.int)

    # Average around the center index.
    halfW = int(w/2)
    centerWindow = img[ centerIdx[0]-halfW : centerIdx[0] + halfW + 1, \
                        centerIdx[1]-halfW : centerIdx[1] + halfW + 1 ]

    centerIntensity = centerWindow.mean()

    # Max r.
    maxR = np.sqrt( centerIdx[0]**2 + centerIdx[1]**2 )

    # x-coordinates of the image.
    xImg = np.linspace(0, img.shape[1]-1, img.shape[1], dtype=np.int)
    xImg = ( xImg - centerIdx[1] ) / maxR

    yImg = centerIntensity / img[centerIdx[0], :]

    # x-axis.
    x = np.linspace( -1, 1, n )
    y = a[0] * x**6 + a[1] * x**4 + a[2] * x**2 + a[3]

    plt.plot( x, y, label="fitted" )
    plt.plot( xImg, yImg, label="raw" )
    plt.xlabel("Dimensionless distance from image center")
    plt.ylabel("Intensity ratio")
    plt.title("Vignetting-correction factor")
    plt.legend()
    plt.savefig(fn)
    plt.show()

if __name__ == "__main__":
    print("Vignetting calibration.")

    parser = argparse.ArgumentParser(description='Vignetting calibration.')
    parser.add_argument("inimage", type=str, help="The input image.")
    parser.add_argument("--window-width", type=int, default=9, help="The window width for evaluating the BGR values.")
    parser.add_argument("--vc-a", type=str, default="vca.dat", help="The filename of the vignetting-correction coefficients.")

    args = parser.parse_args()

    namePart = os.path.splitext( os.path.split(args.inimage)[1] )[0]

    # Load the input image.
    imgOri = cv2.imread( args.inimage, cv2.IMREAD_UNCHANGED )

    if ( 2 != len( imgOri.shape ) ):
        raise Exception("len( imgOri.shape ) == %d." % ( len( imgOri.shape )))

    # Calibrate vignetting mask.
    mask, a = calibrate_vignetting_mask( imgOri, args.window_width )
    print("Vignetting-correction radial response function a =\n{}".format(a))

    # Apply the mask to the white-balanced image.
    imgVC = np.zeros_like( imgOri, dtype=np.uint8 )
    imgVC = cv2.multiply( imgOri, mask, dtype=cv2.CV_8UC1 )

    # Save the vignetting-corrected image.
    fn = namePart + "_VC.png"
    cv2.imwrite( fn, imgVC, [cv2.IMWRITE_PNG_COMPRESSION, 0] )
    print("Vignetting-corrected image saved to %s." % ( fn ))

    # Save the mask as NumPy file and visual image.
    fn = namePart + "_VC_mask.npy"
    np.save( fn, mask )
    print( "Vignetting-correction mask saved as NumPy array to %s." % (fn) )

    visMask = (1 - ( mask - mask.min() ) / ( mask.max() - mask.min() ) ) * 255
    visMask = np.clip( visMask, 0, 255 ).astype( np.uint8 )

    # Save the mask as a visual image.
    fn = namePart + "_VC_mask.png"
    cv2.imwrite( fn, visMask, [cv2.IMWRITE_PNG_COMPRESSION, 0] )
    print("Vignetting-correction mask saved as visual image to %s." % (fn))

    # Save the vignetting-correction coefficients to a text file.
    np.savetxt( args.vc_a, a )
    print("Vignetting-correction coefficients are saved to %s." % ( args.vc_a ))

    # Plot the vignetting-correction curve.
    fn = namePart + "_VC_F.png"
    # plot_vignetting_correction_curve(a, fn = fn)
    plot_vignetting_correction_coefficient_and_image(fn, imgOri, a)

    print("Done.")