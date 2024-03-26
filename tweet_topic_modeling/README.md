# Topic Modeling of Twitter Data using LDA and CorEx

This repository contains code and data for performing topic modeling on some Tweets using Latent Dirichlet Allocation (LDA) and CorEx. The goal is to extract meaningful topics from a collection of tweets and visualize them using various techniques.

## Overview

Topic modeling is a technique used to discover abstract topics within a collection of documents. In this project, the focus is on analyzing a Twitter dataset containing the search term "golden retriever". The aim is to uncover the main themes and discussions present in the tweets.

## Contents

1. **Jupyter Notebook**: The main code is provided in a Jupyter Notebook (`topic_modeling.ipynb`). This notebook contains the Python code for data preprocessing, training an LDA & CorEx models, and visualizing the results. It also includes insights and interpretations of the findings.

2. **Data**: The dataset used in this project is stored in a `.pkl` file (`tweet_data.pkl`). It contains the text of 5000 tweets collected from Twitter's free public API using the search term "golden retriever".

3. **Environment File**: To replicate the environment required to run the code, an `environment.yml` file is provided. This file specifies all the necessary dependencies and installations for the Anaconda environment.

## Usage

To replicate the analysis and visualize the topics:

1. Clone this repository to your local machine.
2. Create a new Anaconda environment using the provided `environment.yml` file:

   ```bash
   conda env create -f environment.yml
   ```
   
3. Activate the newly created environment:

   ```bash
   conda activate tweet_topic_modeling
   ```
   
4. Launch the Jupyter Notebook:

   ```bash
   jupyter notebook tom_robbins_whimsy.ipynb
   ```

5. Follow the instructions in the notebook to execute the code cells and visualize the results.

## Results

The results of the topic modeling analysis are presented in the notebook along with visualizations generated using word clouds, pyLDAvis, and coherence scores. 

