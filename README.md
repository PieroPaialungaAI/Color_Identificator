# Color Identification

## Identify the colors of an image and their position, using Python 

In this repository you will find the script to identify the colors of your image and its specific slice. 
The theoretical background and principles of this work can be found here:

https://towardsdatascience.com/image-color-identification-with-machine-learning-and-image-processing-using-python-f3dd0606bdca]:

An example of the code is the following one
```python
python ColorFinder.py --image_path [path_of_your_image] --color [Color String] --color_number [How many colors you have in your image]
```

And it selects all the squares where the color is dominant wrt the others. 
The application of the code in the example image has reported:
```python
python ColorFinderByString.py --image_path img1.jpg --color '#231806' --color_number 10
```
***
__NOTE:__ If the _exact color_ is not present in the picture the following message will pop out:
```python
The color you have chosen is not one of the dominant colos in this picture

```
Nonetheless, it could be that an extremely near color (e.g. #231806 or #241806) is present. 
***

An example of the code is reported in __results.txt__ , while in __ColorIdentificationUsingML.ipynb__ additional features can be found. 



Happy Rainbow! :) 
