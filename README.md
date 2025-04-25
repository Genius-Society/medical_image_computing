# Chapter II - Classifying AD patients and normal controls from brain images
[![license](https://img.shields.io/github/license/Genius-Society/medical_image_computing.svg)](./LICENSE)
[![hf](https://img.shields.io/badge/huggingface-aal__stats__vol-ffd21e.svg)](https://huggingface.co/spaces/Genius-Society/aal_stats_vol)
[![ms](https://img.shields.io/badge/modelscope-aal__stats__vol-624aff.svg)](https://www.modelscope.cn/studios/Genius-Society/aal_stats_vol)

Classifying AD patients and normal controls from brain images

## Purpose
This project is for AD prediction. Students are expected to be applying what they have learnt in Lab 1 about brain image processing and in Lab 2 about medical image classification to complete this project. 
This is a group-work based project. 

## Data
There are 40 subjects (being either AD patients or normal controls) in the training set and 10 subjects in the test set. The T1-weighted MR brain images for all subjects are provided. In addition, the template images, and the brain AAL atlas could be found under the “Packages” directory.

## Tasks
Predict whether each of the 10 subjects in the test is an AD patient or not. This includes extracting the feature representation from each subject’s MR brain image and classifying these feature vectors using Support Vector Machines.

For feature extraction, students have practiced how to measure the grey matter volume in a single ROI in Lab 1. Now you need to extend the measurement to the first 90 ROIs in AAL atlas for each subject. Therefore, you may do this by using unix shell programming to process these data in batch. A very simple tutorial about unix shell programming is provided in Canvas. Please follow the links in that document for further reading if you need.  The feature extraction process includes four sub-tasks: skullstripping, tissue segmentation (for grey matter), registration, and measurement. To make the tasks easier, we have provided templates of shell scripts to conduct these four sub-tasks, with main steps removed. Please complete those shell scripts for your tasks. It is worth noting that, in those templates, the files are assumed to be organized in a specific directory structure. Be careful to change the directories in the code if you want to use a different directory structure.

<div align=center>    
    <img width="50%" src="https://user-images.githubusercontent.com/20459298/233102075-16faff15-afa4-4a2f-b47a-0441d76ac156.PNG"/><br>
    <b>Figure 1: Folder Arrangement</b>
</div><br>

After feature extraction, each subject is represented by a 90-D feature vector. Please use Support Vector Machine for the classification. You may find the components in the feature vectors have large values and they change largely from feature to feature. You can conduct feature normalization before you apply SVM. For example, you can normalize each feature (across all training samples) to have a zero mean and a unit standard deviation. This may help your classification.

# Report
## Requirements
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## Code download
```bash
git clone -b ad git@github.com:Genius-Society/medical_image_computing.git
cd medical_image_computing
```

## Usage
```bash
python svm.py
```

## Aims & Background
### Aims
This project is for AD prediction: using knowledge of brain image processing and medical image classification to classify AD patients and normal controls from brain images. 
 
There are 40 subjects (AD or normal controls) in the training set and ten issues in the testing set. What we need to do is to pre-process these 50 subjects into a format friendly to medical image classification. Then we need to use the 40 issues processed from the training set to train a classifier and test the performance of the classifier by the ten subjects processed from the testing set.

### Background
AD stands for Alzheimer's disease. It is a progressive, irreversible brain disorder that slowly destroys memory and thinking skills, and eventually the capability to perform the simplest tasks. Compared with a normal brain, the AD brain has many characteristics in shape, such as extreme shrinkage of the cerebral cortex, severely enlarged ventricles and severe shrinkage of the hippocampus. These features can be embodied in brain images so that we can achieve classifications of AD patients and normal controls from them by machine learning.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233102345-9b89a2bb-e91f-42cc-9164-ef4de7383b2d.png"/><br>
<b>Figure 1: Comparison between normal brain and brain affected by AD</b>
</div><br>

Brain image processing performs some operations on a brain image to get an enhanced image or to extract some useful information from it. Most common operations in brain image processing are brain extraction, brain image alignment, brain parcellation and brain tissue segmentation. In this project, brain image processing plays a role of pre-processing for the forward image classification.

## Discussion
There are some different choices that we faced during this project, and we can discuss these several problems.

### How to improve the results of skull stripping
In the skull stripping step, we have found that the initial results of skull stripping are not satisfied. Then we tried two different strategies to deal with this issue. Firstly, we tried to set the central position manually. Then, we used one shell command to finish this process. The advantage of manual operation is of high accuracy. However, it will cost more time to finish this part. The shell command can finish this improvement more quickly with satisfying results.

### Grey matter or grey matter mask
In the measurement step, we use the grey matter mask to calculate the area with the different AAL mask. However, in our first lab, we use the brain area calculated with two AAL masks. Then we think about this issue and try both grey matters and grey matter masks. Finally, we find that the volume result is the same. Therefore, we can use grey matter masks directly to calculate with different AAL masks.

However, the grey matter is not equal to the grey matter mask; this is because the mask only has position information.

### Volume or voxel
In the measurement step, we need to save the area information about different cases and AAL atlas. Here we face a choice of collecting volume values of voxel values. We choose voxel values because different patients may have different brain sizes, which may lead to a massive difference in different brain areas. When we use voxel, we mainly focus on image size, which can receive accurate results.

### Future improvement
During the registration process, we find that the calculation speed is relatively slow, maybe in the future, we can use GPU to help the calculation process FSL to increase the speed.

## Methodology
### FSL
FSL stands for FMRIB Software Library, which is a comprehensive library of analysis tools for FMRI, MRI and DTI brain imaging data. Most of the tools in FSL can be run both from the command line and as GUIs. For example, the FSLeyes, we can browse and process brain images by UI operations on FSLeyes rather than typing command lines. This function is good because we are likely to type wrong commands into a terminal when we want to do some operations, and it helps us avoid boring typing behaviours. In this project, we are using this library to conduct brain image preprocessing, because all the wanted functions among the steps to achieve our goal are integrated into this library. With the combination of Unix Shell commands, we can process brain images in batches. 

### Shell & Python
#### Shell Script
A Shell provides an interface to the Unix system. It gathers input and executes computer programs based on the input. As a program finishes executing, it displays that program's output.

The base concept of a shell script is a list of commands, which are in the order of execution. In this project, the shell script plays an essential role in brain image preprocessing. The batch commands of brain extraction, tissue segmentation, registration and measurement all rely on the shell script. 
 
Shell Script can also define variables; this function not only decreases the error rate of programming but also turns code briefer and more precise. In this project, using variables to replace many prolix file paths is a good option. 

#### Python
Python is a cross-platform programming language, which is a high-level scripting language that combines interpretation, compilation, interactivity and object-oriented. It was initially for writing shells. With the continuous update of the version and the addition of new language features, the more it is used for the development of independent and large-scale projects. 

Besides, Python’s package management, which is a collection of modules, can provide us with functions written by others. For example, the API of the SVM training module is from the third-party group called “sklearn”, and its installation depends on “pip”, which is a syntax from the package management. 

### SVM
SVM stands for Support Vector Machine, it is a generalized linear classifier that performs binary classification of data according to supervised learning, and its decision boundary is the maximum margin for solving the maximum-margin hyperplane. SVMs belong to supervised learning machine learning models widely used for classification and regression tasks.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233102584-ea468900-ec27-4130-860c-f10559b12303.png"/><br>
<b>Figure 2: Schematic diagram of SVM</b>
</div>

### Steps to achieve the goal
#### Skull Stripping
This step aims to extract the brain from raw MRI images because AD features embody in brain shapes. We can use the “bet” command in the FSL environment to achieve brain extraction and browse the result using FSLeyes. The argument “-c” leads the location parameters; if the location is not ideal for view, we will use that argument to refine brain extraction results. 
 
In this project, we need to insert related command lines into the assigned position of the Unix Shell, because the data contains many subjects, which requires batch addressing.

#### Tissue Segmentation
This step aims to segment the brain into grey matters. We can use the “fast” command in the FSL environment to achieve tissue segmentation. Here we need to generate the mask excluding CSF mask and white mask. The related batch commands integrate into Shell Script. The segmentation step will generate many mid products, so using the “rm” command to clean temp files is necessary. The most time cost of one single segmentation operation mainly comes from “fast” command.

#### Registration
This step aims to register the common space to native space, so it is usually called brain image alignment. This procedure can align different brain image slices and find locations of slices with the most similar orientation to each other. It contains two sorts of registration: affine registration and deformable registration. Affine registration is a linear transform, while deformable registration is a non-linear transform. Here shows the difference between the two types of registration: 

<div align=center><b>Table 1: Comparison between affine and deformable registration</b><br>

| Registration type | FSL command | Trandform type | Time cost for single(min) | Generate coeff |
| ----------------- | ----------- | -------------- | ------------------------- | -------------- |
| Affine            | reg_aladin  | Linear         | 15                        | no             |
| Deformable        | reg_f3d     | Non-Linear     | 22                        | yes            |

</div>

Then transform atlas with the “coeff” file generated by deformable registration, to achieve that, we should use the “reg_resample” command in FSL. This procedure also takes a relatively long time. 

#### Measurement
This step aims to measure the volumes of ROIs with the generated masks(transformed atlas). Currently, the command “fslstats” is useful; the inputs of this command are image and mask, respectively. The image awaits to calculate overlay volumes of transformed altas for ROI volumes; these volumes carry shape and position information of the brain, which can reflect the physical status of the tested brain image. Then the volume data formed as a “.csv” file will be used for training the classifier, which is the preparation for AD prediction. From now on, all the shell script works are over. 

#### Classification
This step is to train the SVM classifier and use the trained SVM to make AD predictions, which depends on Python. The grey matter of AD patients has lesions at the designated location. Its classification condition is to learn many AD lesion features at the lesions, and then give a new MRI image to determine whether there are AD lesions at multiple locations of lesions. The whole process is like a regression of a mathematical model, which embodies the essence of machine learning. The computation of accuracy is required to test the performance of a trained classifier.

## Results
### Skull Stripping
In this step, we use the “bet” function to extract the brain image from the whole head. The added code is as follow:

```sh
data_root=$root_path'/Data/'
package_DIR=$root_path'/Packages/'

output_path=$root_path'/Output/'
train_output_path=$root_path'/Output/train/'
test_output_path=$root_path'/Output/test/'

cd $data_root'train/'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing training image "$subID
	fileext='.nii.gz'
	file=$subID
	basename $file $fileext #does not remove suffix, behave like print
	filename="${file%%.*}" #suffix removed
	#skull stripping
	#TODO
	bet $filename $(dirname "$PWD")"/train_brain/"$filename"_brain"$fileext
done

#---- processing test images
cd $data_root'test/'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing test image "$subID
 	fileext='.nii.gz'
     	file=$subID
     	basename $file $fileext
     	filename="${file%%.*}"
	#skull stripping
	#TODO
	bet $filename $(dirname "$PWD")"/test_brain/"$filename"_brain"$fileext
done
```

After this extraction process, we can receive the brain image. However, some results are not satisfying, which shows as below:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233103048-c6858ffb-caf5-46ac-b8e5-7868ee0d3c1b.png"/><br>
<b>Figure 3: Initial extracted brain and head</b>
</div><br>

This extraction result is not satisfied enough; this is because of the incorrect setting of the centre position of the brain. Therefore, we need to figure this issue out, and we tried two different strategies: 

1. Set the central position manually. We can use Fsleyes to acquire the central position of the brain and regenerate the incorrect cases;
2. Use another shell file to align the raw MRI images to its common space.

We tried these two methods together and made the skull stripping results more accurate, which shows as follow:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233103164-74f08d68-e67b-4087-b29e-c0ff2ce3e27d.png"/><br>
<b>Figure 4: Contrast of initial result (green) and improved result (red)</b>
</div><br>

Besides, we run a similar code to finish the skull stripping process of testing data. Then we use the shell file mentioned improving the stripping results. Overall, we finish the first step of skull stripping.

### Tissue Segmentation
In this step, we use the “fast” function to generate the segmentation area of grey matter. Then we use the “mv” function to rename the grey matter segmentation results and use the “rm” function to delete the useless file. The whole changed code shows as below:

```sh
data_root=$root_path'/Data/'
package_DIR=$root_path'/Packages/'

output_path=$root_path'/Output/'
train_output_path=$root_path'/Output/train/'
test_output_path=$root_path'/Output/test/'

cd $data_root'train_brain'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing training image "$subID
     	fileext='.nii.gz'
     	file=$subID
    	basename $file $fileext
     	filename="${file%%.*}"

     	#tissue segmentation
     	base=$train_output_path$filename
     	#TODO
     	fast -S 1 -n 3 -t 1 -g -v -o $filename $filename$fileext
	mv $filename"_seg_1"$fileext ${filename%%%%%%/*}"_greymatter_mask"$fileext
	rm $filename"_pve*"
	fslmaths $filename$fileext -mul ${filename%%%%%%/*}"_greymatter_mask"$fileext $filename"_greymatter"$fileext
     	done


#---- processing test images
cd $data_root'test_brain'
subIDs=`ls`

for subID in $subIDs 
do
echo "Processing test image "$subID

     	fileext='.nii.gz'
     	file=$subID
     	basename $file $fileext
     	filename=$file
     	filename="${file%%.*}"

     	#tissue segmentation
     	base=$test_output_path$filename
     	#TODO
     	fast -S 1 -n 3 -t 1 -g -v -o $filename $filename$fileext
	mv $filename"_seg_1"$fileext ${filename%%%%%%/*}"_greymatter_mask"$fileext
	rm $filename"_pve*"
	fslmaths $filename$fileext -mul ${filename%%%%%%/*}"_greymatter_mask"$fileext $filename"_greymatter"$fileext
done
```

Then we use Fsleyes to see the segmentation result, which shows as follow:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233103286-f633a43d-69c8-4ecc-b1db-d4f8e134bb01.png"/><br>
<b>Figure 5: Segmented Grey matter (yellow)</b>
</div><br>

Also, we run a similar code to finish the grey matter segmentation process of testing data. Overall, we finished the second step of the grey matter segmentation.

### Registration
In this part, we separated the process of code into four steps, which shows as below:

```sh
data_root=$root_path'/Data/'
package_DIR=$root_path'/Packages/'

output_path=$root_path'/Output/'
train_output_path=$root_path'/Output/train_brain/'
test_output_path=$root_path'/Output/test_brain/'
floImg=$package_DIR'MNI152_T1_1mm_brain.nii.gz'
aalImg=$package_DIR'aal.nii.gz'

cd $data_root'Test_brain/'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing test image "$subID
     	refImg=$subID
     	fileext='.nii.gz'
	file=$subID
	basename $file $fileext
	filename="${file%%.*}"
		
	mniAff=$test_output_path$filename"MNI152_T1_native_affine"$fileext
	affTxt=$test_output_path$filename"_MNI2Native_affine.txt"
	mniWarp=$test_output_path$filename"_MNI152_T1_native_warp"$fileext
	warpCoe=$test_output_path$filename"_warpcoeff"$fileext
	trans_aal=$test_output_path$filename"_transformed_aal"$fileext
	
	#affine registration
     	echo "Register MNI template using affine transform"
	#TODO
	reg_aladin -ref $file -flo $floImg -res $mniAff -aff $affTxt
	
	#deformable registration
     	echo "Register MNI template using deformable transform"
	#TODO
	reg_f3d -ref $file -flo $floImg -aff $affTxt -res $mniWarp -cpp $warpCoe

	#transform AAL atlas
    	#TODO
    	reg_resample -ref $file -flo $aalImg -res $trans_aal -trans $warpCoe -inter 0

	#remove files no longer needed
	#TODO
done

#---- process training images
cd $data_root'Train_brain/'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing train image "$subID
     	refImg=$subID
     	fileext='.nii.gz'
	file=$subID
	basename $file $fileext
	filename="${file%%.*}"
	
	mniAff=$train_output_path$filename"MNI152_T1_native_affine"$fileext
	affTxt=$train_output_path$filename"_MNI2Native_affine.txt"
	mniWarp=$train_output_path$filename"_MNI152_T1_native_warp"$fileext
	warpCoe=$train_output_path$filename"_warpcoeff"$fileext
	trans_aal=$train_output_path$filename"_transformed_aal"$fileext
	
	#affine registration
     	echo "Register MNI template using affine transform"
	#TODO
	reg_aladin -ref $file -flo $floImg -res $mniAff -aff $affTxt
	
	#deformable registration
     	echo "Register MNI template using deformable transform"
	#TODO
	reg_f3d -ref $file -flo $floImg -aff $affTxt -res $mniWarp -cpp $warpCoe

	#transform AAL atlas
    	#TODO
    	reg_resample -ref $file -flo $aalImg -res $trans_aal -trans $warpCoe -inter 0

	#remove files no longer needed
	#TODO
done
```

We finished the following tasks:

1. We use reg_aladin function to conduct affine registration to match the MNI template;
2. Then we use the affine transformation matrix to conduct deformable registration to refine the alignment, which is finished by the reg_f3d function;
3. Then we bring the AAL atlas to the native space, by applying the transformation warpcoeff.nii.gz obtained through the above registration process, which is finished by the “reg_resample” function;
4. Finally, we delete the useless intermediate files.

After these four steps, we can receive the transformed AAL atlas overlaid on each case. Here is one example of the transformed AAL atlas in the first case, which shows as follow:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233103361-a0b26b11-79d3-475c-ac7a-64765a5508ab.png"/><br>
<b>Figure 6: Transformed AAL atlas example</b>
</div><br>

Besides, we do a quite similar operation to the test dataset. Overall, we received 50 transformed AAL atlas during this step.

### Measurement
In this step, we used the “fslstats” function to calculate the volume of the coincident area of the grey matter space and each transformed AAL mask. Then we use the “echo -n Value >> filename” function to save the calculated results (Value) to one CSV file in one row. Also, we use the “echo -e Value >> filename” to save the value to the next row. The main code shows as below:

```sh
data_root=$root_path'/Data/'
package_DIR=$root_path'/Packages/'

output_path=$root_path'/Output/'
train_output_path=$output_path'/train_means/'
test_output_path=$output_path'/test_means/'
train_grey_output_path=$output_path'/train_grey/'
test_grey_output_path=$output_path'/test_grey/'
floImg=$package_DIR'MNI152_T1_1mm_brain.nii.gz'
aalImg=$package_DIR'aal.nii.gz'

cd $data_root'train_brain/'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing training image "$subID
     	fileext='.nii.gz'
     	file=$subID
     	basename $file $fileext
     	filename="${file%%.*}"
 
	aal=$filename"_transformed_aal"$fileext
	atlasIMG_DIR=$output_path'train_aal/' 
     	subGMMask=$train_grey_output_path$filename'_greymatter_mask'$fileext

     	echo " >>>> Create measurements for "$filename"  <<<<<<"
     	source ${root_path}/code/CreateSeedMask $atlasIMG_DIR $aal
	echo "i am done!"
     	seedMask_path=$atlasIMG_DIR'seedMasks/'
 	cd $seedMask_path
	mask_IMGs=`ls | wc -l`	
            
     	echo -n $filename >> ${train_output_path}"AAL_statistics_volumn_train.csv"
	for i in $(eval echo {1..$mask_IMGs});
	do
		#echo "i: "${i}
          	#Calculate volume and save it into csv file
          	#TODO
		arr=($(fslstats $subGMMask -k ${i}$fileext -V))
		echo "i: "${arr[0]}
 		echo -n ","${arr[0]} >> ${train_output_path}"AAL_statistics_volumn_train.csv"
     	done	
	#TODO
	echo -e "" >> ${train_output_path}"AAL_statistics_volumn_train.csv"

    	rm '-rf' $seedMask_path   
     	cd $data_root'train_brain/'
done

#----- processing testing images
cd $data_root'test_brain/'
subIDs=`ls`

for subID in $subIDs 
do
	echo "Processing testing image "$subID
     	fileext='.nii.gz'
     	file=$subID
     	basename $file $fileext
     	filename="${file%%.*}"
 
	aal=$filename"_transformed_aal"$fileext
	atlasIMG_DIR=$output_path'test_aal/' 
     	subGMMask=$test_grey_output_path$filename'_greymatter_mask'$fileext

     	echo " >>>> Create measurements for "$filename"  <<<<<<"
     	source ${root_path}/code/CreateSeedMask $atlasIMG_DIR $aal
	echo "i am done!"
     	seedMask_path=$atlasIMG_DIR'seedMasks/'
 	cd $seedMask_path
	mask_IMGs=`ls | wc -l`	
            
     	echo -n $filename >> ${test_output_path}"AAL_statistics_volumn_test.csv"
	for i in $(eval echo {1..$mask_IMGs});
	do
		#echo "i: "${i}
          	#Calculate volume and save it into csv file
          	#TODO
		arr=($(fslstats $subGMMask -k ${i}$fileext -V))
		echo "i: "${arr[0]}
 		echo -n ","${arr[0]} >> ${test_output_path}"AAL_statistics_volumn_test.csv"
     	done	
	#TODO
	echo -e "" >> ${test_output_path}"AAL_statistics_volumn_test.csv"

    	rm '-rf' $seedMask_path   
     	cd $data_root'test_brain/'
done
```

After this code, we can receive one CSV file. In this CSV file, each of 40 training datasets has one name value, 90 voxel value and one label value. Meanwhile, each of 10 testing datasets has one name value, 90 voxel value without label value. There are some parts of the CSV file shown as follow:

<div align=center>
<img width="100%" src="https://user-images.githubusercontent.com/20459298/233103429-3c7c976a-6e7c-42f6-af2b-2c676b5d7875.png"/><br>
<b>Figure 7: The output CSV file</b>
</div><br>

The exact name, voxel and label information shows in the appendix CSV file. It concludes the information of both training and testing datasets. Overall, we generated the required CSV file in this section, which is to finish the final SVM section.

### SVM
In this section, we use python to finish the whole task by dividing the whole process into several steps.

1. We import some necessary packages. Then we create one “norm” function to finish the voxel normalization process. The code of this step shows as below:
```python
import os
from csv import reader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import LinearSVC

def norm(arr):
    x_max = max(arr)
    x_min = min(arr)
    for i in range(len(arr)):
        arr[i] = (arr[i] - x_min) / (x_max - x_min)
    return arr
```

2. Then we separate the 40 training datasets to 30 training datasets and 10 validation datasets because this validation process can let us know the performance of SVM. Then we load training datasets, validation datasets and test datasets one by one, which shows as follow:
```python
# Load training data
with open("data/AAL_statistics_volumn_train.csv") as f:
    csv_data = reader(f, delimiter=',')
    raw_data = np.array(list(csv_data))

# Preprocess training data
x_train = []
y_train = []
data_count = len(raw_data)
tuple_len = len(raw_data[0])

for i in raw_data:
    temp = norm([int(j) for j in i[1:tuple_len - 2]])
    x_train.append(temp)
    if i[tuple_len - 1] == "yes":
        y_train.append(1)
    else:
        y_train.append(0)

# Load validation data
with open("data/AAL_statistics_volumn_val.csv") as f:
    csv_data = reader(f, delimiter=',')
    raw_data = np.array(list(csv_data))

# Preprocess validation data
x_validation = []
y_validation = []
data_count = len(raw_data)
tuple_len = len(raw_data[0])

for i in raw_data:
    temp = norm([int(j) for j in i[1:tuple_len - 2]])
    x_validation.append(temp)
    if i[tuple_len - 1] == "yes":
        y_validation.append(1)
    else:
        y_validation.append(0)

# Load test data
with open("data/AAL_statistics_volumn_test.csv") as f:
    csv_data = reader(f, delimiter=',')
    raw_data = np.array(list(csv_data))

# Preprocess test data
x_test = []
y_test = []
data_count = len(raw_data)
tuple_len = len(raw_data[0])

for i in raw_data:
    temp = norm([int(j) for j in i[1:tuple_len - 1]])
    x_test.append(temp)
```

3. After loading datasets, we use linear SVM to train it through training datasets first.  
Then we use the “predict” function to predict the test results (1 means the AD patient). Finally, we use the “score” function to test the predictability of SVM. The code shows as below:
```python
clf = LinearSVC(loss="hinge", random_state=42).fit(x_train, y_train)
y_test = clf.predict(x_test)
print(y_test)

xt_train, xt_test, yt_train, yt_test = train_test_split(x_train, y_train, test_size=0.33, random_state=73)

#accuracy
clf = LinearSVC(loss="hinge", random_state=42).fit(xt_train, yt_train)
print(clf.score(xt_test, yt_test))
```
```
[0 0 1 0 1 0 1 1 1 1]
1.0
```

From these results, we can find that the 43rd, 45th, 47th, 48th, 49th and 50th people affect AD. Also, the score of our training SVM is 1.0, which is quite good.