## Explanation of the idea
I am conducting fatigue tests of fracture mechanical specimens. SPecifically CT and SEB specimens. 
I need to keep track of the number of cycles and the crack length on each specimen during cyclic loading.
I did not want to allways stop the test, unclamp the specimens, carry them to a remote location where i had a microscope, measure it there and carry it back, clamp it and start the test again. 
Because i needed many data points (Crack_length, number_of_passed_loading_cycles), that would be extremelly time consuming.
What i though of was then to use ArUco markers on the surface of my specimens. I would take images of my specimen during testing and write down the number of cycles.
The markers serve as physical dimensions on the captured images, which allowed me to calculate the location of the crack tip, relative to the ArUco marker. Becuase i imaged the surface of the specimen, together with the aruco marker before a specimen test, i knew teh physical dimensions and could then connect a certain poimt on the image to a physical distance from the marker.

I still needed to stop the test and take images, but that was very fast, as I only had to click a button to make the machine hold its postiion (high frequency caused blurry images as i did not have a fast camera). However it was still many times faster.

## Functionality demands
1. I want to be able to load images that contain ArUco markers, select a position on the image, and obtain the crack length from it using machine vision.
2. I want a live image where i can keep track of the crack tip in real time. This would not be meant for actually savinf the crack lengths.
3. I want to be able to change the parameters that define Aruco marker detection through the UI
4. If possible, i would like a compiled app, so I can transfer it to other computers without having to install python etc

## UI vision
Two "ribbons" or other solution.
The first one contains the live vision functionality
The second one is for loading images and getting crack tip measurements.