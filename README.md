# Udacity-Pipeline

## Table of Contents ##
1. [Installations](#Installations)
2. [Motivations](#Motivations)
3. [File Desriptions](#File_Desc)
4. [Licensing and Acknowledgements](#Licensing)
5. [Results](#Results)

## [Installations](#Installations) ##

This project was created and run using Python version 3.0.

Plugins and imports used were: 
Pandas, PyPlot, SQLite, numpy, Pickle, SKLearn, NLTK

## [Motivations](#Motivations) ##
The point of this project is to analyze messages sent during Natural disasters across the world, which using NLP and ML techniques, will 
be able to classify the data within them. This can help relief organisations with the types of aid needed to be sent to particular locations. 

## [File Desriptions](#File_Desc)
1. App
  - run.py: Flask web app file
  - Templates (Tempalte html files for Flask web app)
    -Master.html
    -Go.html
    
2. Data
  - Disaster.db; Output of ETL Pipeline.
  - Disaster_categories.csv; Disaster dataset for categories of disasters
  - Disaster_messages.csv; Disaster dataset for messages sent during disasters
  - process_data.py; Script used to clean dataframe and ouput a db file
  
3. Models
  - CLassifier.pkl; Trained classifier that is output by the ML Pipeline
  - train_classifier.py; ML pipelines that outputs a classifier file
  
4. ETLPipelinePreparation.ipynb
5. ML Pipeline Preparation.ipynb

### [Licensing and Acknowledgements](#Licensing)

I would like to thank Udacity for their ideas and support for this project. 
I would like to acknowledge FigureEight for the amazing and simple data which they have provided. 

### [Results](#Results)

1. The ETLPipeline was used to clean and transofrm the data. Which was then used as a template to create our Python processing script
2. The ML Pipeline was used to train and output classification categories which was then used as a template to create a scripted pipeline file. 
3. The run.py file can be run to show the results of the pipelines in a flask webapp.


