# A Deep Learning Model for the Identification of Active Contraction Properties of the Myocardium Using Limited Clinical Metrics

  

This repository contains the accompanying code and datasets for the paper "A Deep Learning Model for the Identification of Active Contraction Properties of the Myocardium Using Limited Clinical Metrics". The paper presents a novel approach to use deep learning to assess and forecast the behaviour of an active contraction model of the left ventricular myocardium in a patient-specific clinical setting. Our technique bridges the gap between theoretical calculations and clinical applications, and the repository provides all the necessary resources to replicate our work.

## Authors

*   Igor A. P. Nobrega, Department of Mechanical Engineering, University of South Florida Tampa, FL, USA
*   Wenbin Mao, Department of Mechanical Engineering, University of South Florida Tampa, FL, USA

## Repository Structure

The repository contains several Jupyter notebooks, data files, and trained models. Here is an overview:

*   **PAT\_{number}.ipynb** and **IDEAL\_{number}.ipynb** files: These Jupyter notebooks illustrate the implementation of our model for patient-specific and ideal geometries respectively. The `{number}` represents the step number in the workflow.
*   **pkg** directory: This houses utility functions and other supplementary code required for running the model.
*   **geometrics\_pat** and **geometrics\_ideal** directories: These include the datasets for patient-specific and ideal geometries respectively.
*   **DL\_MODEL\_DS** directory: This folder contains the datasets used for training the deep learning models.
*   **models** directory: This is where we store the trained models.
*   **PVS** directory: This holds the pressure-volume (PV) loops and related data.

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