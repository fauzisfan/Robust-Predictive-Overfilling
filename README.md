# Robust-Predictive-Overfilling

In Virtual Reality, the images will be rendered over the time to make 3D environment in the field of view (FOV). The system will render the image over the time with enough size to over come the black border. However, the proper rendered image size is realy important to avoid extra memory that is produced over the time. So that we need the algorithm so that it can predict the proper image size with eliminating the black border.

On this repo we design the algorithm that is come up with the idea that the image should be rendered larger to overfill the future FOV by using geomatrical transformation. By that calculation, we can be able to design the algorithm more precise by giving the required image size based on the user position.
