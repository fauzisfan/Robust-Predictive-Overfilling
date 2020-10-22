# Robust-Predictive-Overfilling

In Virtual Reality, the images will be rendered over the time to make 3D environment in the field of view (FOV). The system will render the image over the time with enough size to over come the black border. However, the proper rendered image size is realy important to avoid extra memory that is produced over the time. SO that we need the algorithm so that it can predict the proper image size with eliminating the black border.

The algorithm is designed to make the predictive image in Virtual Reality. The method is come up with the idea that the rendered image should be rendered bigger to overfill the future field of view. The algorithm will predict the amount of image that should be rendered over the time. By using geomatrical transformation, we can be able to design the algorithm more precise by giving the required image size based on the user position.
