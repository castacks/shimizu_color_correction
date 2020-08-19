
# Use p3cv4.

import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from Gamma import adjust_gamma

def convert_hex_2_rgb(h):
    """Convert a string of hexadecimal numbers starting with # into BGR values and stored as a list."""

    # Chech if the leading charactar is "#".
    if ( "#" != h[0] ):
        Exception("The leading charactar is not #. h = %s." % (h))

    if ( 7 != len(h) ):
        Exception("The length of the input string is wrong. h = %s." % (h))

    colorCode = h[1:]

    # Conver the hexadecimal values.
    rgb = []
    for i in range(3):
        hexString = colorCode[ i*2:(i*2+2) ]
        rgb.append( int(hexString, 16) )
    
    return rgb

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
    a = np.linalg.solve( RTR, R.transpose().dot( ir ) )

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

if __name__ == "__main__":
    print("Manually white balance.")

    parser = argparse.ArgumentParser(description='Manually white balance am image.')
    parser.add_argument("--input-image", type=str, help="The input image.")
    # parser.add_argument("--target-rgb", type=str, default="#a6a6a6", help="The target rgb value written in hexadecimal form by a leading # symbol.")
    parser.add_argument("--x", type=int, default=-1, help="The column index of the window center. -1 for using the center of the image.")
    parser.add_argument("--y", type=int, default=-1, help="The row index of the window center. -1 for using the center of the image.")
    parser.add_argument("--bf", type=str, default="bf.dat", help="The filename of the balacing factor output file.")
    parser.add_argument("--window-width", type=int, default=9, help="The window width for evaluating the BGR values.")
    parser.add_argument("--avg", action="store_true", default=False, help="Use the average BGR values as the target.")
    parser.add_argument("--vc", action="store_true", default=False, help="Perform vignetting correction.")
    parser.add_argument("--vc-a", type=str, default="vca.dat", help="The filename of the vignetting-correction coefficients.")
    parser.add_argument("--gamma", type=float, default=1.0, help="The Gamma correction coefficient. Use Gamma > 1.0 to make the image brighter.")

    args = parser.parse_args()

    # Load the input image.
    imgOri = cv2.imread( args.input_image, cv2.IMREAD_UNCHANGED )
    
    # Show information of the image.
    print("The size of the image is (%d, %d)." % ( imgOri.shape[0], imgOri.shape[1] ))

    # Get the center pixel index.
    if ( -1 == args.x or -1 == args.y ):
        centerIdx = [ int(imgOri.shape[0] / 2), int(imgOri.shape[1] / 2) ]
    else:
        centerIdx = [ args.y, args.x ]

    # Get the averaged BGR value within the window centered at centerIdx.
    halfWidth = int( args.window_width / 2 )
    centerWindow = imgOri[ centerIdx[0] - halfWidth : centerIdx[0] + halfWidth + 1, \
                           centerIdx[1] - halfWidth : centerIdx[1] + halfWidth + 1, : ]
    
    centerPixel = np.array( [ centerWindow[:, :, 0].mean(), \
                               centerWindow[:, :, 1].mean(), \
                               centerWindow[:, :, 2].mean() ] )

    print("The BGR values of the center pixel is (%d, %d, %d)." % ( centerPixel[0], centerPixel[1], centerPixel[2] ))

    # The target BGR value.
    # targetBGR = convert_hex_2_rgb( args.target_rgb )
    # targetBGR = np.array( targetBGR )

    if ( True == args.avg ):
        # Calculate the average BGR values.
        averageBGR = int(centerPixel.mean())
        targetBGR = np.array([ averageBGR, averageBGR, averageBGR ], dtype=np.int)
        print("Use the average BGR values as the target.")
    else:
        targetBGR = np.array( [ centerPixel[1], centerPixel[1], centerPixel[1] ], dtype=np.int )
        print("Use the green channel as the target.")
    print("The target BGR values are (%d, %d, %d)." % ( targetBGR[0], targetBGR[1], targetBGR[2] ))

    # The balancing factors.
    bf = targetBGR / centerPixel
    print("The balancing factors are: {}".format( bf ))

    imgBalanced = np.zeros_like( imgOri, dtype=np.uint8 )

    # White balance.
    dummyZeroMatrix = np.zeros( [ imgOri.shape[0], imgOri.shape[1] ] , dtype=imgOri.dtype )
    for i in range( 3 ):
        imgBalanced[:, :, i] = cv2.scaleAdd( imgOri[:, :, i], targetBGR[i] / centerPixel[i], dummyZeroMatrix )

    # Save the image.
    namePart = os.path.splitext( os.path.split(args.input_image)[1] )[0]
    fn = namePart + "_Balanced.png"
    cv2.imwrite( fn, imgBalanced )
    print("The balanced image is saved as %s." % ( fn ))

    # Save the balancing factors as a text file.
    np.savetxt( args.bf, bf, fmt="%+e" )
    print("The balancing factors are saved in %s." % ( args.bf ))

    if ( True == args.vc ):
        print("Begin calibrating vignetting effect.")
        # Convert the balanced image into grayscale image.
        gray = cv2.cvtColor( imgBalanced, cv2.COLOR_BGR2GRAY )

        # Calibrate vignetting mask.
        mask, a = calibrate_vignetting_mask( gray, args.window_width )
        print("Vignetting-correction radial response function a =\n{}".format(a))

        # Apply the mask to the white-balanced image.
        imgVC = np.zeros_like( imgBalanced, dtype=np.uint8 )
        for i in range(3):
            imgVC[:, :, i] = cv2.multiply( imgBalanced[:, :, i], mask, dtype=cv2.CV_8UC1 )

        # Save the vignetting-corrected image.
        fn = namePart + "_VC.png"
        cv2.imwrite( fn, imgVC )
        print("Vignetting-corrected image saved to %s." % ( fn ))

        # Save the mask as NumPy file and visual image.
        fn = namePart + "_VC_mask.npy"
        np.save( fn, mask )
        print( "Vignetting-correction mask saved as NumPy array to %s." % (fn) )

        visMask = (1 - ( mask - mask.min() ) / ( mask.max() - mask.min() ) ) * 255
        visMask = np.clip( visMask, 0, 255 ).astype( np.uint8 )

        # Save the mask as a visual image.
        fn = namePart + "_VC_mask.png"
        cv2.imwrite( fn, visMask )
        print("Vignetting-correction mask saved as visual image to %s." % (fn))

        # Save the vignetting-correction coefficients to a text file.
        np.savetxt( args.vc_a, a )
        print("Vignetting-correction coefficients are saved to %s." % ( args.vc_a ))

        # Plot the vignetting-correction curve.
        fn = namePart + "_VC_F.png"
        plot_vignetting_correction_curve(a, fn = fn)

    if ( args.gamma != 1.0 ):
        imgGamma = adjust_gamma( imgBalanced, args.gamma )

        # Save the Gamma corrected image.
        fn = namePart + "_Balanced_Gamma.png"
        cv2.imwrite( fn, imgGamma )
        print("The balanced + gamma corrected image is saved as %s." % ( fn ))