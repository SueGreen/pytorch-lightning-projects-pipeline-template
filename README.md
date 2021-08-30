# Pytorch-lightning-projects-pipeline

## Information about the repository and code
Available tasks:
1. Computer Vision
   1. Image classification

## Running tasks
### Clone the repository
```angular2html
git clone https://github.com/SueGreen/pytorch-lightning-projects-pipeline-template.git
cd pytorch-lightning-projects-pipeline-template
```

### Data
For training an image classification model on a custom dataset, put images in folders with the following structure:
```angular2html
pytorch-lightning-project-pipeline
└───data
    └───test
    │   └───class1_name
    │   │   │   file111.jpg
    │   │   │   file112.jpg
    │   │   │   ...
    │   │
    │   └───class2_name
    │   │   │   file119.jpg
    │   │   │   file120.jpg
    │   │   │   ...
    │   
    └───train
    │   └───class1_name
    │   │   │   file130.jpg
    │   │   │   file131.jpg
    │   │   │   ...
    │   │
    │   └───class2_name
    │   │   │   file140.jpg
    │   │   │   file141.jpg
    │   │   │   ...
```

You can optionally modify a configuration file and then run a task with the following command:
```angular2html
python train.py -c "path/to/config/file"
```


For example, to run a computer vision classification base task, optionally modify "configs/cv_classification_base_config.json" file 
and then specify the path to it:
```angular2html
python train.py -c "configs/cv_classification_base_config.json"
```

