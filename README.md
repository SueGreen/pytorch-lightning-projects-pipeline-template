# Pytorch-lightning-projects-pipeline

## Information about the repository and code
Available tasks:
1. Computer Vision
   1. Image classification (cv_classification_base model)

## Running tasks
### Clone the repository
```angular2html
git clone https://github.com/SueGreen/pytorch-lightning-projects-pipeline-template.git
cd pytorch-lightning-projects-pipeline-template
```

### Data


You can optionally modify a configuration file and then run a task with the following command:
```angular2html
python train.py -c "path/to/config/file"
```


For example, to run a computer vision classification base task, optionally modify "configs/cv_classification_base_config.json" file 
and then specify the path to it:
```angular2html
python train.py -c "configs/cv_classification_base_config.json"
```

To see training process, run ```tensorboard --logdir "checkpoints/<model_name>"```, where model_name is a model from the list.