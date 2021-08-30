# Pytorch-lightning-projects-pipeline

### Data
For training on custom dataset, put images in folders with the following structure:
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

### Running
```angular2html
git clone https://github.com/SueGreen/pytorch-lightning-projects-pipeline-template.git
cd pytorch-lightning-projects-pipeline-template
python train.py -c "configs/cv_classification_base_config.json"
```