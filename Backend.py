import numpy as np
import pandas as pd

import surprise
from surprise import KNNBasic, NMF
from surprise import Dataset, Reader
from surprise import accuracy

import tensorflow as tf
from tensorflow import keras
from keras import layers

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from xgboost import XGBClassifier

from joblib import dump, load
import subprocess

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN",
          "NMF",
          "Neural Network",
          "Regression with Embedding Features",
          "Classification with Embedding Features")


def load_ratings():
    return pd.read_csv("data/ratings.csv")

def load_course_sims():
    return pd.read_csv("data/sim.csv")

def load_courses():
    df = pd.read_csv("data/course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_course_genre():
    df = pd.read_csv("data/course_genre.csv")
    return df

def load_profile():
    df = pd.read_csv("data/profile.csv")
    return df

def load_bow():
    return pd.read_csv("data/courses_bows.csv")

def add_new_ratings(new_courses):
    '''
    This function creates an id for a new user and adds it together
    with selected courses at the end of the rating CSV (user profile).
    '''
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1
        ratings_df = load_ratings()
        new_id = ratings_df['user'].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict['user'] = users
        res_dict['item'] = new_courses
        res_dict['rating'] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("data/ratings.csv", index=False)
        return new_id

# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict

def generate_user_profile():
    '''Dot products of user ratings and course genre values'''
    course_genres_df = load_course_genre()
    ratings_df = load_ratings()
    profile_dict = {}
    user_ids = list(ratings_df.user.unique())
    for user_id in user_ids:
        user_df = ratings_df[ratings_df.user==user_id].sort_values(by='item')
        courses = user_df.item.values
        ratings = user_df.rating.to_numpy()
        genres = course_genres_df[course_genres_df['COURSE_ID'].isin(courses)].sort_values(by='COURSE_ID')
        profile = genres.iloc[:, 2:].to_numpy().T @ ratings
        profile_dict[user_id] = profile
    # Turn the dist to a data frame
    profile_df = pd.DataFrame(profile_dict).T.reset_index()
    profile_df.columns = [['user'] + list(course_genres_df.columns[2:])]
    profile_df.columns = profile_df.columns.get_level_values(0)
    profile_df.to_csv('data/profile.csv', index=False)
    return profile_df

def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    '''
    First find all enrolled courses for a user,
    then find the course similarity values from the sim_matrix.
    '''
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    # Sort the results by similarity
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res  

def user_profile_recommendations(idx_id_dict, enrolled_course_ids, user_id):
    course_genre_df = load_course_genre()
    ratings_df = load_ratings()
    users_ = []
    courses_ = []
    scores_ = []

    # Calculate user profile vector
    user_df = ratings_df[ratings_df.user==user_id].sort_values(by='item')
    courses = user_df.item.values
    ratings = user_df.rating.to_numpy()
    genres = course_genre_df[course_genre_df['COURSE_ID'].isin(courses)].sort_values(by='COURSE_ID')
    profile_vector = genres.iloc[:, 2:].to_numpy().T @ ratings

    # get the unselected course ids for the current user id
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = list(all_courses.difference(enrolled_course_ids))
    unselected_course_df = course_genre_df[course_genre_df['COURSE_ID'].isin(unselected_course_ids)].sort_values(by='COURSE_ID')
    unselected_course_matrix = unselected_course_df.iloc[:, 2:].values

    # user np.dot() to get the recommendation scores for each course
    recommendation_scores = np.dot(unselected_course_matrix, profile_vector)

    # Append the results into the users, courses, and scores list
    for i in range(0, len(unselected_course_ids)):
        score = recommendation_scores[i]
        users_.append(user_id)
        courses_.append(unselected_course_ids[i])
        scores_.append(score)
    return users_, courses_, scores_

# k-means clustering
def kmeans_training(n_clusters=20):
    '''k-means clustering on user profile data and obtain the cluster labels'''
    user_profile_df = generate_user_profile()
    course_genres = user_profile_df.columns[1:]
    scaler = StandardScaler()
    user_profile_df[course_genres] = scaler.fit_transform(user_profile_df[course_genres])
    data = user_profile_df.loc[:, course_genres]
    km = KMeans(n_clusters, random_state=123)
    km.fit_predict(data)
    return km.labels_

def clustering_recommendations(user_id, enrolled_course_ids, n_clusters=20, n_enrollments=10):
    ratings_df = load_ratings()
    profile_df = generate_user_profile()
    user_id_df = profile_df['user']

    labels = kmeans_training(n_clusters)
    label_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_id_df, label_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    user_cluster_label = cluster_df[cluster_df.user==user_id].cluster.values[0]
    
    # Link to courses, and get a list of popular courses in that cluster
    course_cluster_df = pd.merge(ratings_df[['user', 'item']], cluster_df, left_on='user', right_on='user')
    user_course_df = course_cluster_df[course_cluster_df['cluster']==user_cluster_label][['user', 'item']]
    course_enrollment_df = user_course_df.groupby('item').count().reset_index()
    course_enrollment_df.columns = ['course', 'enrollment']
    popular_course_df = course_enrollment_df.loc[course_enrollment_df.enrollment >= int(n_enrollments), :]
    new_course_df = popular_course_df.loc[~popular_course_df.course.isin(enrolled_course_ids), :]
    new_course_df = new_course_df.sort_values(by='enrollment', ascending=False)
    if new_course_df.shape[0] > 0:
        return new_course_df.course.values, new_course_df.enrollment.values

# PCA and k-means clustering   
def pca_training(n_components=9, n_clusters=20):
    '''Perform PCA on user profile data and k-means clustering.
    Obtain the cluster labels.'''
    user_profile_df = generate_user_profile()
    user_id_df = user_profile_df['user']
    course_genres = user_profile_df.columns[1:]

    scaler = StandardScaler()
    user_profile_df[course_genres] = scaler.fit_transform(user_profile_df[course_genres])
    data = user_profile_df.loc[:, course_genres]

    pca = PCA(n_components=9, random_state=123)
    pca = pca.fit(data)
    pca_df = pd.DataFrame(pca.transform(data))  
    pca_df = pd.merge(pd.DataFrame(user_id_df), pca_df, left_index=True, right_index=True)
    pca_df.columns=[['user'] + ['PC'+str(i) for i in range(n_components)]]

    km = KMeans(n_clusters, random_state=123)
    km.fit_predict(pca_df.iloc[:, 1:])
    return km.labels_, pca.explained_variance_ratio_

def pca_recommendations(user_id, enrolled_course_ids, n_components, n_clusters=20, n_enrollments=1):
    # profile_df = load_profile()
    profile_df = generate_user_profile()
    user_id_df = profile_df['user']
    ratings_df = load_ratings()

    labels, _ = pca_training(n_components, n_clusters)
    label_df = pd.DataFrame(labels)
    cluster_df = pd.merge(user_id_df, label_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    user_cluster_label = cluster_df[cluster_df.user==user_id].cluster.unique()[0]
    
    # Link to courses, and get a list of popular courses in that cluster
    course_cluster_df = pd.merge(ratings_df[['user', 'item']], cluster_df, left_on='user', right_on='user')
    user_course_df = course_cluster_df[course_cluster_df['cluster']==user_cluster_label][['user', 'item']]
    course_enrollment_df = user_course_df.groupby('item').count().reset_index()
    course_enrollment_df.columns = ['course', 'enrollment']
    popular_course_df = course_enrollment_df.loc[course_enrollment_df.enrollment >= int(n_enrollments), :]
    new_course_df = popular_course_df.loc[~popular_course_df.course.isin(enrolled_course_ids), :]
    new_course_df = new_course_df.sort_values(by='enrollment', ascending=False)
    if new_course_df.shape[0] > 0:
        return new_course_df.course.values, new_course_df.enrollment.values

# Collaborative filtering with KNN (surprise)
def knn_training(knn_model):
    ratings_df = load_ratings()
    reader = Reader(rating_scale=(2, 3))
    dataset = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)
    # trainset, testset = surprise.model_selection.train_test_split(dataset, test_size=0.25)
    # knn = knn_model.fit(trainset)
    # predictions = knn.test(testset)
    # rmse = accuracy.rmse(predictions)
    trainset = dataset.build_full_trainset()
    knn = knn_model.fit(trainset)
    return knn
    
def knn_recommendations(knn_model, user_id, enrolled_course_ids):
    ratings_df = load_ratings()
    knn = knn_training(knn_model)
    all_courses = ratings_df.item.unique()
    new_courses = np.setdiff1d(all_courses, enrolled_course_ids)
    test_data = [[user_id, course, 3] for course in new_courses]
    predictions = knn.test(test_data)

    user = []
    item = []
    prediction = []
    for i in predictions:
        if i[4]['was_impossible'] == False:
            user.append(i[0])  # uid
            item.append(i[1])  # iid
            prediction.append(round(i[3]))
    knn_df = pd.DataFrame({'user': user,
                           'item': item,
                           'prediction': prediction})
    new_course_df = knn_df.sort_values(by='prediction', ascending=False)
    return new_course_df.item.values, new_course_df.prediction.values

# Collaborative filtering with NMF
def nmf_training():
    ratings_df = load_ratings()
    reader = Reader(rating_scale=(2, 3))
    dataset = Dataset.load_from_df(ratings_df[['user', 'item', 'rating']], reader)
    # trainset, testset = surprise.model_selection.train_test_split(dataset, test_size=0.25)
    trainset = dataset.build_full_trainset()
    nmf = NMF()
    nmf = nmf.fit(trainset)
    return nmf 
    
def nmf_recommendations(user_id, enrolled_course_ids):
    ratings_df = load_ratings()
    nmf = nmf_training()
    all_courses = ratings_df.item.unique()
    new_courses = np.setdiff1d(all_courses, enrolled_course_ids)
    new_course_data = [[user_id, course, 3] for course in new_courses]
    predictions = nmf.test(new_course_data)

    user = []
    item = []
    prediction = []
    for i in predictions:
        if i[4]['was_impossible'] == False:
            user.append(i[0])
            item.append(i[1])
            prediction.append(round(i[3]))
    nmf_df = pd.DataFrame({'user': user,
                           'item': item,
                           'prediction': prediction})
    new_course_df = nmf_df.sort_values(by='prediction', ascending=False)
    return new_course_df.item.values, new_course_df.prediction.values

# Neural network embedding
class RecommenderNet(tf.keras.Model):
    def __init__(self, num_users, num_items, embedding_size=16, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        
        # Define a user_embedding vector
        self.user_embedding_layer = tf.keras.layers.Embedding(
            input_dim=num_users,
            output_dim=embedding_size,
            name='user_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.user_bias = tf.keras.layers.Embedding(
            input_dim=num_users,
            output_dim=1,
            name="user_bias")
        
        # Define an item_embedding vector
        self.item_embedding_layer = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=embedding_size,
            name='item_embedding_layer',
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(1e-6),
        )
        self.item_bias = tf.keras.layers.Embedding(
            input_dim=num_items,
            output_dim=1,
            name="item_bias")
        
    def call(self, inputs):
        '''inputs: user and item one-hot vectors'''
        # Compute the user embedding vector
        user_vector = self.user_embedding_layer(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding_layer(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        # Add all the components (including bias)
        x = dot_user_item + user_bias + item_bias
        # Sigmoid output layer to output the probability
        return tf.nn.relu(x)
    
    def get_config(self):
        base_config = super().get_config()
        config = {
            "num_users": keras.saving.serialize_keras_object(self.num_users),
            "num_items": keras.saving.serialize_keras_object(self.num_items)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        num_users_config = config.pop("num_users")
        num_users = keras.saving.deserialize_keras_object(num_users_config)
        num_items_config = config.pop("num_items")
        num_items = keras.saving.deserialize_keras_object(num_items_config)
        return cls(num_users, num_items, **config)

def process_dataset(rating_df):
    encoded_data = rating_df.copy()
    
    # Mapping user ids to indices
    user_list = encoded_data["user"].unique().tolist()
    user_id2idx_dict = {x: i for i, x in enumerate(user_list)}
    user_idx2id_dict = {i: x for i, x in enumerate(user_list)}
    
    # Mapping course ids to indices
    course_list = encoded_data["item"].unique().tolist()
    course_id2idx_dict = {x: i for i, x in enumerate(course_list)}
    course_idx2id_dict = {i: x for i, x in enumerate(course_list)}

    # Convert original user ids to idx
    encoded_data["user"] = encoded_data["user"].map(user_id2idx_dict)
    # Convert original course ids to idx
    encoded_data["item"] = encoded_data["item"].map(course_id2idx_dict)
    # Convert rating to int
    encoded_data["rating"] = encoded_data["rating"].values.astype("int")
    return encoded_data, user_idx2id_dict, course_idx2id_dict

def prepare_train_test_split(dataset, scale=True):
    # Dataset is one-hot encoded rating_df
    if scale:
        scaler = StandardScaler()
        dataset['rating'] = scaler.fit_transform(dataset['rating'].values.reshape(-1, 1))
    train_set = dataset.sample(frac=0.75, random_state=123)
    val_set_idx = np.setdiff1d(dataset.index.values, train_set.index.values)
    val_set = dataset.iloc[val_set_idx, :]
    X_train = train_set[['user', 'item']].values
    y_train = train_set['rating'].values
    X_val = val_set[['user', 'item']].values
    y_val = val_set['rating'].values
    return X_train, y_train, X_val, y_val

def nn_training(embedding_size, epochs):
    ratings_df = load_ratings()
    num_users = len(ratings_df['user'].unique())
    num_items = len(ratings_df['item'].unique())
    encoded_data, _, _ = process_dataset(ratings_df)
    X_train, y_train, X_val, y_val = prepare_train_test_split(encoded_data)

    model = RecommenderNet(num_users, num_items, embedding_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', 
                                                    metrics=[tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(X_train, y_train,
          epochs=epochs, batch_size=32, shuffle=True,
          validation_data=(X_val, y_val))
    model.save(f'nn_embedding_{epochs}epochs.keras')
    return model, history

def nn_recommendations(user_id, enrolled_course_ids, embedding_size, epochs):
    ratings_df = load_ratings()
    encoded_data, _, _ = process_dataset(ratings_df)
    # model, _ = nn_training(embedding_size, epochs)
    model = keras.models.load_model(f'nn_embedding_{epochs}epochs.keras')

    all_courses = ratings_df.item.unique()
    new_courses = np.setdiff1d(all_courses, enrolled_course_ids)
    new_course_df = ratings_df[ratings_df.item.isin(new_courses)]
    new_encoded_df = encoded_data.loc[new_course_df.index, ['user', 'item']]
    new_encoded_data = new_encoded_df.values
    new_encoded_data[:, 0] = new_encoded_data[:, 0].max()  # user_idx
    new_course_ratings = model.predict(new_encoded_data)
    subprocess.run(['rm', f'nn_embedding_{epochs}epochs.keras'])
        
    train_set = encoded_data.sample(frac=0.75, random_state=123)
    y_train = train_set['rating'].values
    scaler = StandardScaler()
    scaler.fit(y_train.reshape(-1, 1))

    new_course_ratings = scaler.inverse_transform(new_course_ratings)
    new_course_df['rating'] = new_course_ratings
    new_course_df = new_course_df.sort_values(by='rating', ascending=False)
    new_course_df['rating'] = new_course_df['rating'].round()
    return new_course_df.item.values, new_course_df.rating.values

# Regression with Embedding Features
def generate_embedding_features(embedding_size, epochs):
    ratings_df = load_ratings()
    user_df = pd.DataFrame(ratings_df.user.unique())
    item_df = pd.DataFrame(ratings_df.item.unique())
    nn_model, _ = nn_training(embedding_size, epochs)

    user_embedding = nn_model.get_layer('user_embedding_layer').get_weights()[0]
    item_embedding = nn_model.get_layer('item_embedding_layer').get_weights()[0]

    user_emb_df = pd.DataFrame(user_embedding)
    user_emb_df = pd.merge(user_df, user_emb_df, left_index=True, right_index=True)
    user_emb_df.columns = [['user'] + [f'UFeature{i}' for i in range(embedding_size)]]
    user_emb_df.columns = user_emb_df.columns.get_level_values(0)

    item_emb_df = pd.DataFrame(item_embedding)
    item_emb_df = pd.merge(item_df, item_emb_df, left_index=True, right_index=True)
    item_emb_df.columns = [['item'] + [f'IFeature{i}' for i in range(embedding_size)]]
    item_emb_df.columns = item_emb_df.columns.get_level_values(0)

    user_emb_merged = pd.merge(ratings_df, user_emb_df, how='left', on='user').fillna(0)
    merged_df = pd.merge(user_emb_merged, item_emb_df, how='left', on='item').fillna(0)

    u_features = user_emb_df.columns[1:]
    i_features = item_emb_df.columns[1:]
    user_embeddings = merged_df[u_features]
    course_embeddings = merged_df[i_features]
    ratings = merged_df['rating']

    embedding_dataset = user_embeddings + course_embeddings.values
    embedding_dataset.columns = [f"Feature{i}" for i in range(embedding_size)]
    embedding_dataset['rating'] = ratings
    # embedding_dataset.to_csv('embedding_dataset.csv', index=False)
    return embedding_dataset, user_emb_df, item_emb_df

def regression_embedding_training(lm_model, embedding_size, epochs):
    embedding_dataset, _, _ = generate_embedding_features(embedding_size, epochs)
    X = embedding_dataset.iloc[:, :-1]
    y = embedding_dataset.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    lm_model.fit(X_train, y_train)
    y_pred = lm_model.predict(X_test)
    rmse = np.sqrt(((y_test - y_pred)**2).mean())
    out_label = str(lm_model).split('(')[0]
    dump(lm_model, f'{str(out_label)}_emb_trained.joblib') 
    return lm_model, rmse

def get_new_user_embeddings(user_id, enrolled_course_ids, embedding_size, epochs):
    ratings_df = load_ratings()
    embedding_dataset, user_emb_df, item_emb_df = generate_embedding_features(embedding_size, epochs)

    # Get the embedded vector for the new user
    new_user_emb = user_emb_df.iloc[-1, 1:].values
    # Get the embedded item vector for unselected courses
    all_courses = ratings_df.item.unique()
    new_courses = np.setdiff1d(all_courses, enrolled_course_ids)
    new_course_df = ratings_df[ratings_df.item.isin(new_courses)].drop('rating', axis=1)
    new_course_df['user'] = user_id
    new_course_df = new_course_df.drop_duplicates(subset='item', ignore_index=True)
    new_course_emb_df = item_emb_df[item_emb_df.item.isin(new_courses)]

    # Add user embedded vector to item embedded vector
    embeddings = new_course_emb_df.iloc[:, 1:].values + new_user_emb
    return new_course_df, embeddings

def regression_embedding_recommendations(lm_model, user_id, enrolled_course_ids, embedding_size, epochs):
    new_course_df, embeddings = get_new_user_embeddings(user_id, enrolled_course_ids, embedding_size, epochs)
    # model, _ = regression_embedding_training(lm_model, embedding_size, embedding_size, epochs)
    model_label = str(lm_model).split('(')[0]
    model = load(f'{str(model_label)}_emb_trained.joblib')
    ratings = model.predict(embeddings)
    subprocess.run(['rm', f'{str(model_label)}_emb_trained.joblib'])
    subprocess.run(['rm', f'nn_embedding_{epochs}epochs.keras'])

    new_rating_df = pd.DataFrame(ratings)
    new_course_df['rating'] = new_rating_df.iloc[:, 0]
    new_course_df.columns = ['user', 'item', 'rating']
    new_course_df = new_course_df.sort_values(by='rating', ascending=False)
    new_course_df['rating'] = new_course_df['rating'].round()
    return new_course_df.item.values, new_course_df.rating.values

# Classification with Embedding Features
def classification_embedding_training(classifier, embedding_size, epochs):
    '''
    Classifiers: LogisticRegression(), RandomForestClassifier(), XGBClassifier(),
    with opitmal hyperparameters.
    '''
    embedding_dataset, _, _ = generate_embedding_features(embedding_size, epochs)
    encoder = LabelEncoder()
    X = embedding_dataset.iloc[:, :-1]
    y = embedding_dataset.iloc[:, -1]
    y = encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    training_report = accuracy, precision, recall, f1_score
    out_label = str(classifier).split('(')[0]
    dump(classifier, f'{out_label}.joblib')
    return classifier, training_report

def classification_embedding_recommendations(classifier, user_id, enrolled_course_ids, embedding_size, epochs):
    ratings_df = load_ratings()
    new_course_df, embeddings = get_new_user_embeddings(user_id, enrolled_course_ids, embedding_size, epochs)
    # clf, _ = classification_embedding_training(classifier, embedding_size, epochs)
    clf_label = str(classifier).split('(')[0]
    clf = load(f'{clf_label}.joblib')
    predicted_labels = clf.predict(embeddings)
    subprocess.run(['rm', f'{clf_label}.joblib'])
    subprocess.run(['rm', f'nn_embedding_{epochs}epochs.keras'])
    # Transform (0, 1) back to (2, 3)
    encoder = LabelEncoder()
    encoder.fit(ratings_df['rating'])
    new_course_ratings = encoder.inverse_transform(predicted_labels)
    new_course_df['rating'] = new_course_ratings
    new_course_df = new_course_df.sort_values(by='rating', ascending=False)
    new_course_df['rating'] = new_course_df['rating'].round()
    return new_course_df.item.values, new_course_df.rating.values

# ------------------------------------------------------------------
# Model training
def train(model_name, params):
    # Similarity and Profile
    if model_name == models[0] or model_name == models[1]:
        pass

    # Clustering
    if model_name == models[2]:
        n_clusters = params['n_clusters']
        kmeans_training(n_clusters)

    # Clustering with PCA
    if model_name == models[3]:
        n_components = params['n_components']
        n_clusters = params['n_clusters']
        pca_training(n_components, n_clusters)

    # KNN (surpise)
    if model_name == models[4]:
        sim_measure = params['sim_measure']
        content = params['content']
        if content == 'item':
            sim_options = {'name': sim_measure, 'user_based': False}
        else:
            sim_options = {'name': sim_measure, 'user_based': True}
        knn_model = KNNBasic(sim_options=sim_options)
        knn_training(knn_model)

    # NMF (surprise)
    if model_name == models[5]:
        nmf_training()

    # NN
    if model_name == models[6]:
        embedding_size = params['embedding_size']
        epochs = params['epochs']
        nn_training(embedding_size, epochs)

    # Regression with embedding features
    lm_models = {'Ridge': Ridge(alpha=1.0, solver='saga', random_state=123, max_iter=1000),
                'Lasso': Lasso(alpha=0.3, random_state=123),
                'ElasticNet': ElasticNet(random_state=123)}
    if model_name == models[7]:
        lm_model_label = params['lm_model']
        lm_model = lm_models[lm_model_label]
        embedding_size = params['embedding_size']
        epochs = params['epochs']
        regression_embedding_training(lm_model, embedding_size, epochs)

    # Classification with embedding features
    clf_models = {'LogisticRegression': LogisticRegression(random_state=123),
                'RandomForestClassifier': RandomForestClassifier(n_estimators=10, max_depth=30, random_state=123),
                'XGBClassifier': XGBClassifier(n_estimators=30, max_depth=25, learning_rate=0.2, random_state=123)}
    if model_name == models[8]:
        classifier_label = params['classifier']
        classifier = clf_models[classifier_label]
        embedding_size = params['embedding_size']
        epochs = params['epochs']
        classification_embedding_training(classifier, embedding_size, epochs)
    
    else:
        pass

# -----------------------------------------------------------------
# Prediction
def predict(model_name, user_ids, params):    
    idx_id_dict, id_idx_dict = get_doc_dicts()
    ratings_df = load_ratings()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        user_ratings = ratings_df[ratings_df['user'] == user_id]
        enrolled_course_ids = user_ratings['item'].to_list()

        # Course Similarity
        if model_name == models[0]:
            sim_threshold = 0.6
            if "sim_threshold" in params:
                sim_threshold = params["sim_threshold"] / 100.0
            sim_matrix = load_course_sims().to_numpy()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, 
                                                    enrolled_course_ids, sim_matrix)
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)

        # User profile
        if model_name == models[1]:
            profile_threshold = params['profile_threshold']
            users_, courses_, scores_ = user_profile_recommendations(idx_id_dict, 
                                                                    enrolled_course_ids, user_id)
            for idx, score in enumerate(scores_):
                if score >= profile_threshold:
                    users.append(users_[idx])
                    courses.append(courses_[idx])
                    scores.append(round(score))               

        # Clustering
        if model_name == models[2]:
            n_clusters = params['n_clusters']
            n_enrollments = params['n_enrollments']
            new_courses, enrollments = clustering_recommendations(user_id, enrolled_course_ids, 
                                                                  n_clusters, n_enrollments)
            if len(new_courses) > 0:
                for i in range(len(new_courses)):
                    users.append(user_id)
                    courses.append(new_courses[i])
                    scores.append(enrollments[i])

        # Clustering with PCA
        if model_name == models[3]:
            n_components = params['n_components']
            n_clusters = params['n_clusters']
            n_enrollments = params['n_enrollments']
            new_courses, enrollments = pca_recommendations(user_id, enrolled_course_ids, 
                                                           n_components, n_clusters, n_enrollments)
            ### bug: will get None type error if no data returned
            if len(new_courses) > 0:
                for i in range(len(new_courses)):
                    users.append(user_id)
                    courses.append(new_courses[i])
                    scores.append(enrollments[i])

        # KNN (surprise)
        if model_name == models[4]:
            predicted_rating = params['predicted_rating']
            sim_measure = params['sim_measure']
            content = params['content']
            if content == 'item':
                sim_options = {'name': sim_measure, 'user_based': False}
            else:
                sim_options = {'name': sim_measure, 'user_based': True}
            knn_model = KNNBasic(sim_options=sim_options)
            new_courses, predicted_ratings = knn_recommendations(knn_model, user_id, enrolled_course_ids)
            if len(new_courses) > 0:
                for i in range(len(predicted_ratings)):
                    if predicted_ratings[i] >= predicted_rating:
                        users.append(user_id)
                        courses.append(new_courses[i])
                        scores.append(predicted_ratings[i])

        # NMF
        if model_name == models[5]:
            predicted_rating = params['predicted_rating']
            new_courses,  predicted_ratings = nmf_recommendations(user_id, enrolled_course_ids)
            if len(new_courses) > 0:
                for i in range(len(predicted_ratings)):
                    if predicted_ratings[i] >= predicted_rating:
                        users.append(user_id)
                        courses.append(new_courses[i])
                        scores.append(predicted_ratings[i])

        # NN
        if model_name == models[6]:
            predicted_rating = params['predicted_rating']
            embedding_size = params['embedding_size']
            epochs = params['epochs']
            new_courses, predicted_ratings = nn_recommendations(user_id, enrolled_course_ids, 
                                                                embedding_size, epochs)
            if len(new_courses) > 0:
                for i in range(len(predicted_ratings)):
                    if predicted_ratings[i] >= predicted_rating:
                        users.append(user_id)
                        courses.append(new_courses[i])
                        scores.append(predicted_ratings[i])

        # Linear regression with embedding features
        if model_name == models[7]:
            lm_model = params['lm_model']
            predicted_rating = params['predicted_rating']
            embedding_size = params['embedding_size']
            epochs = params['epochs']  
            new_courses, predicted_ratings = regression_embedding_recommendations(lm_model, user_id, enrolled_course_ids, 
                                                                                embedding_size, epochs)
            if len(new_courses) > 0:
                for i in range(len(predicted_ratings)):
                    if predicted_ratings[i] >= predicted_rating:
                        users.append(user_id)
                        courses.append(new_courses[i])
                        scores.append(predicted_ratings[i])

        # Classification with embedding features
        if model_name == models[8]:
            classifier = params['classifier']
            predicted_rating = params['predicted_rating']
            embedding_size = params['embedding_size']
            epochs = params['epochs']  
            new_courses, predicted_ratings = classification_embedding_recommendations(classifier, user_id, enrolled_course_ids, 
                                                                                embedding_size, epochs)
            if len(new_courses) > 0:
                for i in range(len(predicted_ratings)):
                    if predicted_ratings[i] >= predicted_rating:
                        users.append(user_id)
                        courses.append(new_courses[i])
                        scores.append(predicted_ratings[i])

    if 'top_courses' in params: 
        top_courses = params['top_courses']                 
        users = users[:int(top_courses)]
        courses = courses[:int(top_courses)]
        scores = scores[:int(top_courses)]

    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    return res_df
