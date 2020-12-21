import streamlit as st

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import requests
from io import BytesIO


@st.cache
def cluster_image(array, n_centroids=3, seed=1):
    # shape must be (x, y, 3)
    assert len(array.shape) == 3
    assert array.shape[-1] == 3
    
    height, width, _ = array.shape
    X = array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_centroids, random_state=seed).fit(X)

    labels = kmeans.predict(X)
    new_image = np.zeros((labels.shape[0], 3))
    # convert to int so they're interpreted as 255 RGB
    centroids = kmeans.cluster_centers_.astype(int)
    for i in range(labels.shape[0]):
        new_image[i] = centroids[labels[i]]

    return new_image.reshape(height, width, 3).astype(int)

@st.cache
def get_image(url):
    r = requests.get(url)
    return np.asarray(Image.open(BytesIO(r.content)))


st.title('Color Clustering')

image_url = st.text_input('Enter an image URL', 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.DlSCO0LVdUaJ-6lvmNcZggHaEd%26pid%3DApi&f=1')
centroids = st.number_input(
    'Enter the number of colors you want to cluster into',
    min_value=2,
    max_value=50,
    value=3
)

original_container, clustered_container = st.beta_columns(2)
with original_container:
    st.subheader('Original Image')
    orig_image_data = get_image(image_url)
    original_image = st.image(orig_image_data, use_column_width=True)
with clustered_container:
    st.subheader('Clustered Image')
    clustered_image_data = cluster_image(orig_image_data, centroids)
    clustered_image = st.image(clustered_image_data, use_column_width=True)


