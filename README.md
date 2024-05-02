### A recommender app for online courses ###

This app was an optional part of the final project of an [IBM machine learning course](https://www.coursera.org/professional-certificates/ibm-machine-learning). The instructors provided the data and initial codes setting up the app structures (see the original scripts directory), and many of the functions were in lab practices. However, it still demanded considerable efforts to put various pieces together into one program and to make sure it worked properly.     

The app was not intended to be a practical system; it was written to complete the course in due time. It is a tool to show how the system works and the competence of a learner. Most of the algorithms need some time to run. In my current setup, many of the recommendation steps also run the training algorithms. To speed up the recommendation step, some trained models will be saved to the disk. Writing privilege is required.     

The app was developed in a virtual environment with Python 3.11 with numpy and pandas packages. Other dependencies are:    

1. streamlit
2. streamlit-aggrid
3. sklearn
4. xgboost
5. tensorflow
6. joblib  

To try out the app in old way, download the whole package, create an virtual env and install the required pacakges. Then, on command line, run:  

```
streamlit run recommender_app.py
```

This will open the interface.

Be patient and have fun! :)