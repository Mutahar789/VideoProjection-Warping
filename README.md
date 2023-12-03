# VideoProjection-Warping
Python project to project video clips onto static images using affine and projective transformations, with added realism through alpha blending.

## Usage
- Install OpenCV: `pip install opencv-python`
- Run the main script: `python main.py`

## Methodology
- Video Splitting: Splits videos into frames at a specified frame rate.
- Point Reading: Reads corresponding points between images for transformation.
- Self Warping: Applies estimated transformation matrices to images. 
- Video Projection: Projects video clips onto static images, simulating them on screens.
- Alpha Blending: Adds realism through alpha blending techniques.

Note: All image warping, homography, and affine transformation algorithms in this project are implemented from scratch.

