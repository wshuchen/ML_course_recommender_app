import numpy as np
import pandas as pd
import time
import Backend as Backend

import streamlit as st
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode


# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)


# ------- Functions ------
# Load datasets
@st.cache_data
def load_ratings():
    return Backend.load_ratings()

@st.cache_data
def load_course_sims():
    return Backend.load_course_sims()

@st.cache_data
def load_courses():
    return Backend.load_courses()

@st.cache_data
def load_course_genre():
    return Backend.load_course_genre()

@st.cache_data
def load_profile():
    return Backend.load_profile()

@st.cache_data
def load_bow():
    return Backend.load_bow()


# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_genre_df = load_course_genre()
        # profile_df = load_profile()
        course_bow_df = load_bow()

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):
    if model_name in Backend.models:
        with st.spinner('Training...'):
            time.sleep(0.5)
            Backend.train(model_name, params)
        st.success('Done!')
    else:
        pass


def predict(model_name, user_ids, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = Backend.predict(model_name, user_ids, params)
    if len(res) >= 1:
        st.success('Recommendations generated!')
    else:
        st.write('No course meet your criteria.')
        st.write('Please adjust your criteria and try again.')
    return res


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')

# Initialize the app
selected_courses_df = init__recommender_app()
params = {}

# Course number display slider
st.sidebar.subheader('1. Select top courses to display')

# Add a slide bar for selecting top courses
top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=100,
                                    value=10, step=1)
params['top_courses'] = top_courses

# Model selection selectbox
st.sidebar.subheader('2. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    Backend.models
)

# Hyper-parameters for each model
# params = {}
st.sidebar.subheader('3. Tune Hyper-parameters: ')

# Course similarity model
if model_selection == Backend.models[0]:
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['sim_threshold'] = course_sim_threshold

# User profile model
if model_selection == Backend.models[1]:
    profile_threshold = st.sidebar.slider('Cousre recommendation Score Threshold',
                                             min_value=0, max_value=50,
                                             value=2, step=1)
    params['profile_threshold'] = profile_threshold

# Clustering model
elif model_selection == Backend.models[2]:
    n_clusters = st.sidebar.slider('Number of clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    n_enrollments = st.sidebar.slider('Number of enrollments in a course',
                                   min_value=10, max_value=10000,
                                   value=50, step=20)
    params['n_clusters'] = n_clusters
    params['n_enrollments'] = n_enrollments

# Clustering with PCA
elif model_selection == Backend.models[3]:
    n_components = st.sidebar.slider('Number of PCA components',
                                   min_value=1, max_value=14,
                                   value=9, step=1)

    n_clusters = st.sidebar.slider('Number of clusters',
                                   min_value=0, max_value=50,
                                   value=20, step=1)
    n_enrollments = st.sidebar.slider('Number of enrollments in a course',
                                   min_value=10, max_value=10000,
                                   value=50, step=20)
    params['n_components'] = n_components
    params['n_clusters'] = n_clusters
    params['n_enrollments'] = n_enrollments

# KNN
elif model_selection == Backend.models[4]:
    predicted_rating = st.sidebar.slider('predicted_rating',
                                             min_value=2, max_value=3,
                                             value=3, step=1)
    # params['top_courses'] = top_courses
    sim_measure = st.sidebar.selectbox('Select a similarity measure:',
                                ('MSD', 'cosine', 'pearson', 'pearson_baseline'))
    content = st.sidebar.selectbox('Select the content for calculation:',
                                ('user', 'item'))
    params['predicted_rating'] = predicted_rating
    params['sim_measure'] = sim_measure
    params['content'] = content

# NMF
elif model_selection == Backend.models[5]:
    predicted_rating = st.sidebar.slider('predicted_rating',
                                             min_value=2, max_value=3,
                                             value=3, step=1)
    # params['top_courses'] = top_courses
    params['predicted_rating'] = predicted_rating

# NN
elif model_selection == Backend.models[6]:
    predicted_rating = st.sidebar.slider('predicted_rating',
                                             min_value=2, max_value=3,
                                             value=3, step=1)
    epochs = st.sidebar.slider('Epochs',
                               min_value=5, max_value=30,
                               value=10, step=1)
    embedding_size = st.sidebar.slider('Embedding size',
                                       min_value=16, max_value=32,
                                       value=16, step=2)
    # params['top_courses'] = top_courses
    params['predicted_rating'] = predicted_rating
    params['epochs'] = epochs
    params['embedding_size'] = embedding_size

# Regression with embedding features
elif model_selection == Backend.models[7]:
    lm_model_selection = st.sidebar.selectbox(
                                'Select a regression model:',
                                ('Ridge', 'Lasso', 'ElasticNet'))
    predicted_rating = st.sidebar.slider('predicted_rating',
                                         min_value=2, max_value=3,
                                         value=3, step=1)
    epochs = st.sidebar.slider('Epochs',
                               min_value=5, max_value=30,
                               value=10, step=1)
    embedding_size = st.sidebar.slider('Embedding size',
                                        min_value=16, max_value=32,
                                        value=16, step=2)
    params['lm_model'] = lm_model_selection
    params['predicted_rating'] = predicted_rating
    params['epochs'] = epochs
    params['embedding_size'] = embedding_size

# Regression with embedding features
elif model_selection == Backend.models[8]:
    classification_model = st.sidebar.selectbox(
                                'Select a classification model:',
                                ('LogisticRegression', 'RandomForestClassifier', 'XGBClassifier'))
    predicted_rating = st.sidebar.slider('predicted_rating',
                                         min_value=2, max_value=3,
                                         value=3, step=1)
    epochs = st.sidebar.slider('Epochs',
                               min_value=5, max_value=30,
                               value=10, step=1)
    embedding_size = st.sidebar.slider('Embedding size',
                                        min_value=16, max_value=32,
                                        value=16, step=2)
    params['classifier'] = classification_model
    params['predicted_rating'] = predicted_rating
    params['epochs'] = epochs
    params['embedding_size'] = embedding_size

else:
    pass


# Training
st.sidebar.subheader('4. Training: ')
training_button = st.sidebar.button("Train Model")
training_text = st.sidebar.text('')
# Start training process
if training_button:
    train(model_selection, params)
    # if model_selection == Backend.models[3]:
    # #     ev_ratio = train(model_selection, params)
    # #     st.write('PCA cumulative explained variance ratio:', np.cumsum(ev_ratio))
    # # else:
        # train(model_selection, params)


# Prediction
st.sidebar.subheader('5. Prediction')

# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    new_id = Backend.add_new_ratings(selected_courses_df['COURSE_ID'].values)
    user_ids = [new_id]
    res_df = predict(model_selection, user_ids, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"]).drop('COURSE_ID', axis=1)

    if model_selection == Backend.models[2] or model_selection == Backend.models[3]:
        res_df.columns = ['ENROLLMENT', 'TITLE', 'DESCRIPTION']
    if model_selection in [Backend.models[i] for i in [4, 5, 6, 7, 8]]:
        res_df.columns = ['RATING', 'TITLE', 'DESCRIPTION']
        res_df['RATING'] = res_df['RATING'].astype(int)
    res_df = res_df.drop_duplicates(subset='TITLE', ignore_index=True)

    st.table(res_df)
