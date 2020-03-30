# Pipeline-2: Modeling 
This is the process of image classification and preparation for AWS lambda deployment.
The goal of classification is to detect map features of interest in image.

### Map Features
The map features are grouped according to color, shape, and texture patterns,
* _**color and shape**_ : Features with specifically defined color and general shapes 
    (e.g. Highway (motorway, trunk, primary, …), railroad)
* _**color**_ : Features with specific color and undefined shape 
    (e.g. Aeroway-runway, landuse-commercial)
* _**template**_ : A single isolated template (e.g Helipad, hospital, fire stations, etc.)

* _**texture**_ :  Features of specific color with texture and undefined shape 
    (e.g. landuse-military, landuse-quarry, etc.)
    
    
### Classification Techniques
1.	**CNN**\
The model is particularly effective to learns shapes (edges) and colors.\
_CNN_1_model_training.py_ creates _tf_model.pb_ in _tf-models_ directory under the current working directory 
given train_data_dir, validation_data_dir, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir,
epochs, and batch_size.\
_CNN_2_trigger_function.py_ is for AWS lambda trigger function.

2.	**Color-based Feature Detection**\
Use the distribution of pixel colors used in each tile for a positive dataset. 
The mode of the color distribution of positive data set is a ‘dominant color’ of the feature 
which characterizes the class. 
It can be used only for the features that use unique colors to represent area or or lands.\
_COLOR_1_dominant_colors.py_ creates a list of n-most RGB color dominant in the featured map tiles.\
_COLOR_2_model.py_ is for AWS lambda trigger function.

3.	**Template Matching**\
Template Matching is a method for searching and finding the location of a template image in a larger image.
Python libarary ‘OpenCv’ provides 6 different template matching methods based on distance measure between 
template and patches of the image. To use this technique effectively, all templates that we look for should be provided. 
For example, it we want to capture ‘fire station’ even partial features, 
we need to prepare partial images of” fire station” as template images.\
_Temple_Match_model.py_ is for AWS lambda trigger function given the path of the directory 
where template images are.

4. **Local Binary Pattern (to be continued)**
It looks the neighbor pixels of each pixel and convert them into binary (0 or 1) thresholding 
the center pixel to describe texture. The histogram of these binaries characterizes the overall pattern 
of the image. One approach we can think of is that we superimpose a grid over each image and 
compare LBP histograms of patches with the LBP histogram of the featured tile (e.g. military). 
If any patch with the LBP histogram similar enough that of the featured tiles detected, 
we may predict the image as a ‘military’ area.    
_reference_: \
http://www.scholarpedia.org/article/Local_Binary_Patterns\
https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/


### Data Directory Structure for modeling
##### 1. CNN
```bash
├── TEST
│   ├── negative
│   └── positive
└── TRAIN
    ├── negative
    └── positive
```
##### 2. Color-Based Feature Detection
```bash
.
├── positive_sample (to determine featured color of map feature)
├── test-negative
└── test-positive
```

##### 3. Template Match
```bash
.
├── templates
├── test-negative
└── test-positive
```