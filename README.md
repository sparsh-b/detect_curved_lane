# Detection of curved lane & Prediction of its Radius of Curvature
- This is done as part of project2 of ENPM673 at UMD.
- First, the region of interest is cropped & the homography of the image is computed & the warped image is calculated.
- Then, binary threshold is applied. This process is able to isolate the divider lines.
- Then, the histogram of the resultant image is calculated so that approximate position of the lanes is derived.
- Then, a sliding window search is applied to detect the pixels corresponding to the lanes.
- Here, a window is initialized for each lane to the positions given by the histogram.
- Then, the number of white pixels (corresponding to the lanes) inside the window is calculated & if this number is above a threshold, the window in the next iteration will be moved to the centroid of the white pixels of the current iteration.
- This is continued until the ends of the image are reached.
- The pixels falling within the window at each iteration are considered as the final pixels of each of the lanes.
- Then, np.polyfit() is used to calculate the approximate 2nd degree polynomial which fits each of the detected lanes & the radius of curvature is caculated.
- Usage: `python3 turn_predict.py`. 
- Output is written to the directory `results.`

## Results
- The below video shows the outpt of the algorithm.
- The top part of the video shows the curved lane detected.
- The bottom-right shows the detected solid & dashed divider lines.
- The bottom left shows the homography (top-view) of the divider lines.

https://user-images.githubusercontent.com/45140884/220531403-4f32b385-a8e3-4884-9d3c-8252bc128238.mp4

