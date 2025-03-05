# Fall Detection

## Motivation
Falls are the most common cause of injury to elderly, and thus I thought it would be neat to use cheap ESP32 Cameras to detect falls. 

## Method
The first step is to make use of my hip space to 3D space calibration documented [here](https://matthew-bird.com/blogs/Hip-to-Camera-Space.html) to allow me to use the faster [mediapipe pose detection model](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) to get the 3D body joint positions around the camera as the origin. 

The next step is to match the bodies detected in each frame to each other. This is done by calculating the relative velocity between each existing body and each body detected in the current frame in a matrix of shape (# bodies 1, # bodies 2), and then taking the minimum value in the matrix to match 2 bodies, repeating until all the bodies are matched. If the velocity threshold is exceeded, which would happen in case one person walks off screen to the right and another appears on screen to the left, it would define the body detected in the current frame as a new body instead of matching it. 

The last step is to use the change in velocity between each of the joints of a body to calculate the acceleration each joint underwent in the last frame, and if the acceleration of any joint was exceeded, report an incident and save a video of the incident. 

The code itself can be used on any camera as a result of its automatic calibration of the camera view depth, though at the same time it’s important to have a clear view of a person to calibrate the camera with at the start of the video. The code also prints the camera view depth so it can be inputted as a parameter to the code when run using the same camera in the future, reducing overhead runtime on startup substantially. 

## Setup
1. Download pose landmarker [here](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python) and rename to pose_landmarker.task 
2. Download model [here](https://bit.ly/metrabs_l) and rename folder to model
3. Download .mp4 files to the In directory
4. Run example.ipynb
