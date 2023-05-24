# A Deep Learning Model for the Identification of Active Contraction Properties of the Myocardium Using Limited Clinical Metrics

  
This repository contains the accompanying code and datasets for the paper "A Deep Learning Model for the Identification of Active Contraction Properties of the Myocardium Using Limited Clinical Metrics". The paper presents a novel approach to use deep learning to assess and forecast the behaviour of an active contraction model of the left ventricular myocardium in a patient-specific clinical setting. Our technique bridges the gap between theoretical calculations and clinical applications, and the repository provides all the necessary resources to replicate our work.

## Authors

*   Igor A. P. Nobrega, Department of Mechanical Engineering, University of South Florida Tampa, FL, USA
*   Wenbin Mao, Department of Mechanical Engineering, University of South Florida Tampa, FL, USA

## Abstract

  

Technological breakthroughs have enhanced our understanding of myocardial mechanics and physiological responses to detect early disease indicators. Using constitutive models to represent myocardium structure is critical for understanding the intricacies of such complex tissues. Several models have been developed to depict both passive response and active contraction of myocardium; however they require careful adjustment of material parameters for patient-specific scenarios and substantial time and computing resources. Thus, most models are unsuitable for employment outside of research. Deep learning (DL) has sparked interest in data-driven computational modeling for complex system analysis. We developed a DL model for assessing and forecasting the behavior of an active contraction model of the left ventricular (LV) myocardium under a patient-specific clinical setting. Our original technique analyzes a context in which clinical measures are limited: as model input, just a handful of clinical parameters and a pressure-volume (PV) loop are required. This technique aims to bridge the gap between theoretical calculations and clinical applications by allowing doctors to use traditional metrics without administering additional data and processing resources. Our DL model's main objectives are to produce a waveform of active contraction property that properly portrays patient-specific data during a cardiac cycle and to estimate fiber angles at the endocardium and epicardium. Our model accurately represented the mechanical response of the LV myocardium for various PV curves, and it applies to both idealized and patient-specific geometries. Integrating artificial intelligence with constitutive-based models allows for the autonomous selection of hidden model parameters and facilitates their application in clinical settings.

  

**Keywords**: _Cardiac mechanics; Active contraction of myocardium; Constitutive modeling; Deep learning; Material parameter estimation; Finite element analysis_

  

* * *

## Repository Structure

The repository contains several Jupyter notebooks, data files, and trained models. Here is an overview:

*   **PAT\_{number}.ipynb** and **IDEAL\_{number}.ipynb** files: These Jupyter notebooks illustrate the implementation of our model for patient-specific and ideal geometries respectively. The `{number}` represents the step number in the workflow.
*   **pkg** directory: This houses utility functions and other supplementary code required for running the model.
*   **geometrics\_pat** and **geometrics\_ideal** directories: These include the datasets for patient-specific and ideal geometries respectively.
*   **DL\_MODEL\_DS** directory: This folder contains the datasets used for training the deep learning models.
*   **models** directory: This is where we store the trained models.
*   **PVS** directory: This holds the pressure-volume (PV) loops and related data.

## _Note_

_When executing the Jupyter notebooks, the user may need to adjust the directory names and paths according to their specific configuration. This code was originally developed on Google Drive and used with Google Colab. If you are running the code locally, you might need to make necessary adaptations._

## How to Use

To use this repository, you should first clone it to your local machine. You can do this by using the following command in your terminal:

```plain
bash
git clone https://github.com/<username>/DL-for-the-Identification-of-Active-Contraction-of-LV--public
```

Replace `<username>` with the username of the repository owner.

Next, you can navigate to the cloned repository by using the following command:

```plain
bash
cd DL-for-the-Identification-of-Active-Contraction-of-LV--public
```

Once you're in the repository, you can start exploring the different directories and running the Jupyter notebooks. We recommend starting with the notebooks `PAT_1.ipynb` or `IDEAL_1.ipynb` to understand the general workflow.

## Requirements

This project requires Python 3.7 or later and the following Python libraries installed:

*   [NumPy](http://www.numpy.org/)
*   [Pandas](http://pandas.pydata.org/)
*   [Matplotlib](http://matplotlib.org/)
*   [Scikit-Learn](http://scikit-learn.org/stable/)
*   [TensorFlow](https://www.tensorflow.org/)

You also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html).

## Contact

If you have any questions or comments about the code, please feel free to reach out to us via email or through GitHub.