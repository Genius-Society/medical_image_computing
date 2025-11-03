# Chapter I - Medical image computing (MIC)
[![license](https://img.shields.io/github/license/Genius-Society/medical_image_computing.svg)](./LICENSE)

Medical Image Computing Module Development

## Intro
MIC is an interdisciplinary field at the intersection of computer science, information engineering, electrical engineering, physics, mathematics, and medicine. This field develops computational and mathematical methods for solving medical images' problems and their use for biomedical research and clinical care. Within the MIC domain, there are three essential medical image analysis techniques: Medical Image Enhancement (MIE) and Medical Image Segmentation (MIS).

## Requirements
- [3D Slicer](https://download.slicer.org)
- [Sample data](./data)

## Tasks
### MIE
You are expected to program an image filtering algorithm with Python, which performs a convolution on the 3D volume _MRHead.nrrd_. The filters to be used, including the smoothing, sharpening, and edge detection filters. 

1.	Load data _MRHead.nrrd_ to Slicer;
2.	Import the source code _MedicalImageEnhancement.py_ to Slicer. Then restart Slicer, and find _Task A - MIE_ module in _Assignment_;
3.	Open source code _MedicalImageEnhancement.py_;
4.	After modifying your code, save it and then click on the _Reload_ button to reload the module, so you don’t need to restart Slicer;
<div align=center><img width="605" src="https://user-images.githubusercontent.com/20459298/233113671-eafaeb7d-562e-4d31-9927-d255b8bd29fe.PNG"/></div>

5.	Change the layout to displace Red Slice only. Superimpose the MRHead onto MRHead_filtered, and then change the opacity to see the difference between them.
<div align=center><img width="605" src="https://user-images.githubusercontent.com/20459298/233113730-e387c90c-a6de-4b78-b731-97e607cba13a.png"/></div>

### MIS
Region-growing algorithms can perform medical image segmentation tasks via delineating ROIs iteratively. 

1.	Load _MRBrainTumor.nrrd_ from the files provided. Use the _Editor_ module to draw a single dot in the slice which tumor has a clear boundary;
<div align=center><img width="605" src="https://user-images.githubusercontent.com/20459298/233113835-c93019a4-4dd3-4577-b768-3b50de5e8708.png"/></div>

2.	Import the source code _MedicalImageSegmentation.py_ to Slicer. Then restart Slicer, and find _Task B - MIS_ module in _Assignment_;
3.	Open source code _MedicalImageSegmentation.py_;
4.	After modifying your code, save it and then click on the _Reload_ button to reload the module, so you don’t need to restart Slicer;
5.	Click _Apply_ to see the results. Tune the global and local parameters to find the best segmentation result.

## Report
Based on experiment results, the best global and local parameters are among experiments with ID from 2 to 3 considering whether the classification is true or false, positive or negative. The best way to find them is to define an indicator of the best result, then define a distance between the result indicator in current parameters and the best outcome, and finally search for the settings of minimum distance by deep learning. In that way, we do not need to search for the best parameters manually.

### MIE
The source code of MIE is in _MedicalImageEnhancement.py_.

_Table 1_ compares the three filters, including smoothing, sharpening, edge detection in the aspects of the kernel, experiment demo, and time cost of running. The way of achieving a filter is to use the filter to perform convolution operations on 3D images.

<div align=center>
    <b>Table 1: Comparison of different filters</b><br>
    <img width="455" src="https://user-images.githubusercontent.com/20459298/233113143-f6f0b426-17f5-4b38-8d13-d7ae4af03a9a.PNG"/>
</div>

### MIS
The source code of MIS is in _MedicalImageSegmentation.py_.

_Figure 1_ shows the demonstration of 3D segmentation results achieved using three multiple viewing angles.

<div align=center>    
    <img width="100%" src="https://user-images.githubusercontent.com/20459298/233113218-15beccc4-0f79-4cea-a283-9e3956de3ee2.png"/><br>
    <b>Figure 1: 3D segmentation result of the tumor</b>
</div><br>

_Table 2_ shows the demonstration of the experiments on different global and local parameter combinations. 

<div align=center>
    <b>Table 2: Part experiments on different global and local parameters</b><br>
    <img width="455" src="https://user-images.githubusercontent.com/20459298/233113328-43869ae5-f62f-42bd-9808-a836d714089a.PNG"/>
</div><br>