# Chapter III - Classifying Cell Images Using Deep Learning Models
[![license](https://img.shields.io/github/license/Genius-Society/medical_image_computing.svg)](https://github.com/Genius-Society/medical_image_computing/blob/hep2/LICENSE)
[![hf](https://img.shields.io/badge/huggingface-HEp2-ffd21e.svg)](https://huggingface.co/spaces/Genius-Society/HEp2)
[![ms](https://img.shields.io/badge/modelscope-HEp2-624aff.svg)](https://www.modelscope.cn/studios/Genius-Society/HEp2)

Classifying Cell Images Using Deep Learning Models

## Purpose
This project is to exercise how to use deep CNN models to classify cell images. Efficient Human Epithelial-2 (HEp-2) cell image classification can facilitate the diagnosis of many autoimmune diseases.

## Data
The HEp-2 cell images are from the international competition on cell image classification hosted by International Conference on Pattern Recognition in 2014. The images have been pre-partitioned into training set (8701 images), validation set (2175 images), and test set (2720 images). In addition, a .csv file is enclosed. It contains the category of each image. This file consists of two columns: the first column is the image IDs of all the 13,596 images, and the IDs are consistent with the names of the images in the three sets; and the second column is the category of the cell image.

## Tasks
Please finetune the pretrained AlexNet model for the cell image classification task. Specifically, you need to:

1) use the architecture of AlexNet to construct the cell image classification model;
2) use the weights in the pretrained AlexNet to initialize the weights of the model;
3) change the number of neurons in the output layer (i.e., number of classes) in the model from 1000 (used by imageNet)  to 6 classes (used in the task);
4) Train the model for a few epochs;
5) Use the finetuned model to classify the test cell images.

# Report
## Requirements
```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install -r requirements.txt
```

## Usage
### Code download
```bash
git clone -b hep2 git@github.com:Genius-Society/medical_image_computing.git
cd medical_image_computing
```

### Train
```bash
python train.py
```

### Draw training curves
```bash
python plotter.py
```
It will automatically find the latest log to plot.

## Results
| ![](https://www.modelscope.cn/models/Genius-Society/HEp2/resolve/master/loss.jpg) | ![](https://www.modelscope.cn/models/Genius-Society/HEp2/resolve/master/acc.jpg) |
| :-------------------------------------------------------------------------------: | :------------------------------------------------------------------------------: |
|                                  **Loss curve**                                   |                       **Training and validation accuracy**                       |

## Aims & Background
### Aims
This project aims to classify cell images using deep CNN models. Our task consists of five steps:

1)	Construct our cell image classification model using AlexNet framework; 
2)	Initialise the weights in the pre-trained AlexNet for our model; 
3)	Modify the number of neurons in the output layer in our model into six classes; 
4)	Train our model for a few epochs; 
5)	Classify the test cell images using the fine-tuned model.
 
The training set contains 8,701 images, the validation set includes 2,175 cell images, and the test set contains 2,720 cell images. Besides, there is a label file containing ID and category of the overall 13,596 cell images. These are all our experiment materials from the international competition on cell image classification hosted by the International Conference on Pattern Recognition in 2014 (ICPR2014).

<div align=center><b>Table 1: The usage of three sorts of dataset</b><br></div>

| Dataset type   | Usage                                                                                                                                                                                  |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Training set   | They are data samples used for model fitting;                                                                                                                                          |
| Validation set | It is a set of samples set aside separately during model training used to adjust the hyperparameters of the model and to conduct a preliminary evaluation of the model's capabilities; |
| Test set       | It can evaluate the generalization ability of the final model but not as a basis for algorithm-related selection such as parameter tuning and feature selection.                       |

### Background
The human epithelial 2 (HEp-2) cells are epithelial cells of carcinoma of the larynx human. They are used for laboratory diagnostics for the detection of autoimmune antibodies and antinuclear antibodies. This specific type of cells represents the substrate of choice for the search for autoimmune antibodies with the indirect immunofluorescence technique due to their tumour nature.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233107373-c7518fcd-2fae-4e6d-9979-1679345df4cd.png"/><br>
<b>Figure 1: HEp-2 example cell images of ICPR2014 dataset</b>
</div><br>

Efficient image classification of HEp-2 cells can facilitate the diagnosis of many autoimmune diseases. That is why we make a classification for HEp-2 cells. However, image classification is a heavy workload and requires high accuracy. Manual search is unrealistic; traditional image segmentation algorithms are not enough to meet the demand. Therefore, the use of a deep CNN-based image segmentation algorithm is more appropriate.

## Discussion & Conclusion
### Discussion
Transfer learning: we can use the pretrained model and change some part of it to improve the training performance. This is because this pretrained model has trained in large dataset such as ImageNet. Therefore, this model has a stronger learning ability. In addition, we need to save the model parameters within the training process in order to evaluate it and retrain this task.
 
Overfitting: due to the small dataset of our input images, the overfitting problem will become the most effective issue. The best way to decrease the influence of overfitting is using the “Dropout” function during the fully connected layer.
 
Learning rate: according to our experiment, the large learning will study fast in the beginning. However, the large learning rate will easily lead training accuracy to not change too much after several epochs. In addition, too small a learning rate still cannot receive better performance. Therefore, we can use schedule function to change learning from large to small to receive better performance.

### Conclusion
In conclusion, we finish the dataset establishing process, model establishing process and model training and evaluating process. We finally received a test accuracy of 92% using the following parameters:

<div align=center><b>Table 2: Experiment parameters</b><br>

| Parameter | epoch | iteration | batch_size |
| :-------: | :---: | :-------: | :--------: |
|   Value   |  40   |    10     |     4      |

</div>

### Future work
We need to fine-tune the topology structure of AlexNet network to find more accurate parameters for the application scenario of HEp-2 cell image classification but finding the network structure most adaptive for this application scenario requires a lot of experiments. It is a task requiring a considerable workload. To improve search efficiency, using auto-machine learning to explore more optimised network structure is an efficient option. That is to use a small part of the data set to test the network performance under a specific topological structure, convolution kernel size and other parameters. When we obtain the locally optimal solution after multiple trials, then we can use the extensive data set for training.
 
We can also try other CNN frameworks as the base network to fine-tune structure and look for a more accurate method for HEp-2 cell image classification. No matter from the aspect of accuracy or time cost, any small optimisation is an improvement. The brief orientations for labour focus on the classifier, activation functions, loss functions, topology structure of convolutional layers, etc. For example, we can change the FC layer into SVM as the classifier and compare the performance of the two kinds of the classifier. We can also change Cross-entropy loss function into smooth L1 loss and make a comparison. Besides, we can change the current activation function into Mish function and make a comparison.

## Methodology
The main technical methods used in the whole experiment are listed below:

### Python
Python is a cross-platform programming language, as well as a high-level scripting language that combines interpretation, compilation, interactivity and object-oriented, which was initially designed by shell coders. With the continuous update of the version and the addition of new language features, the more it is used for the development of independent and large-scale projects.

Python’s package management, which is a collection of modules, can provide us with functions written by others. For example, the API of the CSV module implements the reading and writing of form data in CSV format; its mean syntax is as follow:

```python
import csv 
 with open('gt_training.csv',"rt", encoding="utf-8") as csvfile: 
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
    for row in rows:
        ...
```

### Deep Learning
Deep learning is a sort of AI function that mimics the way the human brain worked when processing data used to detect objects, recognise speech, translate language and make decisions. Deep learning AI can learn unsupervised from unstructured and unlabelled data. The process of deep learning is equivalent to the operation of regression. A neural network is a model that contains a large number of undetermined parameters. The learning process is to fill in these uncertain parameters. This process needs to refer to the backpropagation algorithm.
 
The backpropagation algorithm works by calculating the gradient of the loss function for each weight through a chain rule, one slope at a time, iterating backwards from the last level to avoid redundant calculation of intermediate terms in the chain rule. The steps to perform a deep learning model is as follow:
 
1)	Design layer structures;
2)	Set up deep learning layers;
3)	Define loss function and optimiser;
4)  Train the model with training data;
5)  Predict the test data.

### CNN
The CNN stands for the convolutional neural network, which is a kind of feedforward neural network that includes convolution computing and has a fully connected system. As one of the usual methods in deep learning, CNN has representative learning capabilities and can do the shift-invariant classification of input information due to its hierarchical structure, so they are also named shift-invariant artificial neural networks.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233107716-956f513f-f866-4160-bf12-d8f7523451c3.png"/><br>
<b>Figure 2: The architecture of original CNN</b>
</div><br>

The training steps of CNN model are as below:
1)	Input the training data into the model;
2)	Obtain the result from the current model;
3)	Calculate the loss with loss functions;
4)	Gain gradients by backpropagating the loss;
5)	Update model weights with gradient vectors and pre-defined learning rate.

### Pytorch
PyTorch is an open-source library for Python machine learning, based on Torch, C++ implements the bottom layer, and it plays an essential role in AI fields, such as natural language processing. It was initially developed by Facebook’s AI research team and used in Uber’s probabilistic programming software Pyro. PyTorch has two main features: Tensor calculation like NumPy, which can be accelerated by GPU;
 
To install it via pip, we can use the command line generator on Pytorch official index (https://pytorch.org/get-started/locally/) to automatically create the installation command line and copy it into the terminal of our system. Directly using ‘pip install pytorch’ or ‘pip3 install --upgrade alexnet_pytorch’ may cause pip to return errors.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233107811-bd66a2e9-ddc9-42d1-920d-810313250912.png"/><br>
<b>Figure 3: The command line generator on Pytorch official website</b>
</div>

### AlexNet
AlexNet took part in the ImageNet Large-Scale Visual Recognition Challenge on 30 September 2012. The network's top 5 errors were 15.3%, 10.8 percentage points lower than the second place. The main result of the original paper was that the depth of the model was critical to its high performance and that the model was computationally expensive but feasible due to the use of a GPU in the training process.

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233107927-487cb63c-4895-44b8-9007-bda42ae1c661.png"/><br>
<b>Figure 4: The structure of the AlexNet</b>
</div><br>

The steps to finetune the pre-trained AlexNet on Pytorch are as follow: 

1)	Construct the cell image classification model; 
2)	Initialise the weights of the model from the pre-trained model; 
3)	Modify class number to six; 
4)	Train the model with preferred epochs; 
5)	Justify the accuracy of the model by test sets.

### Data Augmentation
Data augmentation is like an image pre-processing in the training phase. When our dataset is not enough for training, and we cannot find more images to supplement our dataset, data augmentation is a good way to expand our dataset. This process is like the practice questions for students are not enough, change the existing questions and continue to train the students. There are many data augmentation methods, such as:

1) Crop, rotation, flip, hue, saturation, exposure, aspect;
2) MixUp;
3)	CutMix;
4)	Mosaic;
5) Blur;

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233108002-d4be2fe8-8bea-4e62-a1f8-c6accd52b3d0.png"/><br>
<b>Figure 5: All kinds of data augmentation methods</b>
</div><br>

But we need to pay attention to maintaining the correctness of the answer when performing data augmentation, or our model would learn the wrong knowledge.

## Results
### Dataset loading and pre-processing
In this section, we finish the dataset loading and pre-processing task.
 
First, we want to use the “ImageFolder” function to finish the load the dataset and finish the transform process. Therefore, due to the limitation of “ImageFolder” function, we must put all train, validation and test images to different folders that use their label name to name the file name. Therefore, we can use the package from “csv”, “shutil” and “os” to move images to the folder which correspond to their labels.

We use the “csv.reader” function to read the csv file which has the information of the images’ file names and labels. Then we use the “os.path.exist” function to search the filename and follow the first row in the csv file. Finally, we use the “shutil.move” function to move the files to their corresponding files. Overall, we finish the pre-process of using “ImageFolder” function.

Then we use pytorch function to load and move dataset to dataloader of model.

We use the “transforms.Compose” function to conclude several augmentation functions that we use in this task. For example, using the “Resize” to change the image size. This is because the initial image size is small which cannot provide enough information for the image classification process. In addition, we use the “CenterCrop” to cut image to square input due to the demand of the AlexNet model. The “Normalize” function is to finish the normalization process and can decrease the contrast issue of initial data. Furthermore, the “RandomAffine” function is used to increase the amount and randomicity of input data.

Then we use the “DataLoader” function to make the dataset to several load groups. The batch size is used to control the number of datasets in one loader group. According to the experiment experience, too small batch size will lead to the fluctuating training process, which is hard to train. In addition, too large batch size will increase the memory of GPU and lead to the decline of model generalization ability. Therefore, we apply batch size to 4 in this task.

### Deep learning model
Then we develop the AlexNet model and the whole code is shown below:

```python
model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

for parma in model.parameters():
    parma.requires_grad = False
          
model.classifier = torch.nn.Sequential(nn.Dropout(),
                                       nn.Linear(256 * 6 * 6, 4096),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(),
                                       nn.Linear(4096, 4096),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.5),
                                       nn.Linear(4096, 1000),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(1000,6))
```

First we use the “torch.hub.load” function to load the pretrained AlexNet. Then due to the classification target of this task being 6, we must change the fully connected layers. We must use the “require_grad=False” function to set the returned grad to none. This is because the pretrained model has their own grad which we do not know. In addition, the change of model architecture will also lead to the change of grad. Therefore, in order to change some parts of the pretrained net, we want to freeze some parts of the model and cannot return the grad again. So, we have to set the returned grad to “False”. Then we rewrite the classifier layer of the AlexNet and add one dropout layer to avoid the overfitting, one activation function (“Relu”) to increase non-linear and one linear layer to change the final output channels to finish the classify task. We can use the “model.eval()” to evaluate the whole model, which is shown as follow:

```txt
AlexNet(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace=True)
    (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace=True)
    (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace=True)
    (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(6, 6))
  (classifier): Sequential(
    (0): Dropout(p=0.5, inplace=False)
    (1): Linear(in_features=9216, out_features=4096, bias=True)
    (2): ReLU(inplace=True)
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=4096, out_features=4096, bias=True)
    (5): ReLU(inplace=True)
    (6): Dropout(p=0.5, inplace=False)
    (7): Linear(in_features=4096, out_features=1000, bias=True)
    (8): ReLU(inplace=True)
    (9): Linear(in_features=1000, out_features=6, bias=True)
  )
)
```

We can see that the final output channels are changed from 1,000 to 6.

### Training and evaluation process
Then we set some basic functions of the training process such as loss function, optimizer function, learning rate schedule and GPU CUDA setting. The whole code is shown below:

```python
import torch.optim as optim
lr=0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.classifier.parameters(), lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=lr, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

torch.cuda.empty_cache()
model=model.cuda()
criterion = criterion.cuda()
for state in optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()
```

We use the cross-entropy loss in this task. This is because this cross-entropy loss has been approved to perform well in image classification tasks. Then we use the “SGD” optimizer to optimize the model. It is important that we only change the parameters of the classifier layer, this is because the pretrained model already trained well for the initial convolutional layer. Then we use the “ReduceLROnPlateau” function to decrease the learning rate to one tenth if the loss hasn't changed too much for 5 epochs. This function requires us to input the loss to finish. Finally, we put model, data and criterion to GPU to increase the training speed.

Then we develop the training and evaluation function of this task, which is shown as follow:

```python
epoch_num=40
iteration=10

for epoch in range(epoch_num):  # loop over the dataset multiple times
    epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
    print(f'{epoch_str:-^40s}')
    print(f'Learning rate: {optimizer.param_groups[0]["lr"]}')
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data[0].cuda(),data[1].cuda()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % iteration == iteration - 1:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / iteration))
            loss_list.append(running_loss / iteration)
        running_loss = 0.0
    eval_model_train(model, trainloader, tra_acc_list)            
    eval_model_validation(model, validationloader, val_acc_list)
    scheduler.step(loss.item())

print('Finished Training')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def eval_model_train(model, trainLoader, tra_acc_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of trainloader: %d %%' % (100 * correct / total))
    tra_acc_list.append(100 * correct / total) 
    
def eval_model_validation(model, validationLoader, val_acc_list):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validationLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of validationloader: %d %%' % (100 * correct / total))
    val_acc_list.append(100 * correct / total)
```

```txt
cuda:0
```

For the training process, we first get the input data and move them to GPU. Then we use this model to train the input image and receive output results. After this process, we use loss function to calculate the loss by predicting output and ground truth label and move this loss backwards for the optimizer function. After several iterations, we collect the average loss in order to draw the loss curve. In addition, we run evaluation functions to evaluate the training results by accuracy. In the final of one epoch training, we use the schedule function to change the learning rate for the next epoch.

For the evaluation process, we input the image and use the model to finish the classification process and receive one classification result. Then we compare this predicted result with the ground truth label to calculate the accuracy. We need to evaluate the training dataset accuracy and validation dataset accuracy in the training process. The main purpose of collecting validation accuracy is to avoid overfitting.

Then we use the “matplot” function to plot the loss curve and accuracy curve. The code is shown below:

```python
import matplotlib.pyplot as plt
import numpy as np

def show_point(max_id, list):
    show_max='['+str(max_id+1)+' '+str(list[max_id])+']'
    plt.annotate(show_max, xytext=(max_id+1, list[max_id]), xy=(max_id+1, list[max_id]))

plt.figure(1)
plt.axis([0, 260, 0, 0.01])
plt.plot(iter_list, loss_list, label="loss", color="red", linestyle="-", linewidth=1)
plt.xlabel("validation iteration")
plt.ylabel("loss")
plt.title("loss curve")
plt.legend()
plt.show()

x_acc=[]
for i in range(len(tra_acc_list)):
    x_acc.append(i+1)

x=np.array(x_acc)
y1=np.array(tra_acc_list)
y2=np.array(val_acc_list)
max1=np.argmax(y1)
max2=np.argmax(y2)
plt.title('Accuracy of training and validation')
plt.xlabel('Epoch')
plt.ylabel('Acc(%)')
plt.plot(x, y1, label="Training")
plt.plot(x, y2, label="Validation")
plt.plot(1+max1, y1[max1], 'r-o')
plt.plot(1+max2, y2[max2], 'r-o')
show_point(max1, y1)
show_point(max2, y2)
plt.legend()
plt.show()
```

Then we can receive the loss curve and show it in the following picture:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233108330-9b671a37-c4c7-409f-a305-9a58e3b58a87.png"/><br>
<b>Figure 6: Loss curve</b>
</div><br>

Although the loss is fluctuating during the whole process, the main trend of this curve is in decreasing.

Then we plot the accuracy curve of training and validation data loader, which is shown as follow:

<div align=center>
<img width="605" src="https://user-images.githubusercontent.com/20459298/233108425-a9268ec2-1334-4643-acc8-bf81e4099003.png"/><br>
<b>Figure 7: Training and validation accuracy</b>
</div><br>

We can find that the training and validation accuracy are increasing at beginning and levelling off after about 20 epochs. In addition, the accuracy curve of training and validation dataset are separated at about 20 epochs, this is because of the influence of overfitting.

Then we test the accuracy on the test loader and get 92% accuracy, which is shown below:

```python
def eval_model_test(model, testLoader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of test: %d %%' % (100 * correct / total))

eval_model_test(model, testloader)
```

```txt
Accuracy of test: 92 %
```