import streamlit as st
import tensorflow as tf
import visualkeras

MODEL_PATH = 'model_CNN/model.h5'


# @st.cache
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

def CNN_predict(file):
    # TODO: implement a function that recognizes the recording in the file using the tensorflow model
    pass

model = load_model(MODEL_PATH)

st.markdown('# Speech command Recognition')
st.image(visualkeras.layered_view(model, legend=True))

input, output = st.columns(2, gap='large')
with input:
    st.markdown('## Input')
    uploaded_file = st.file_uploader("Upload your recording here: ", type=['wav'])
    st.markdown('## Preview')
    if uploaded_file is not None:
        # To read file as bytes:
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')
        
with output:
    st.markdown('## Output')
    
# st.image(ann_viz(model, view=True, filename='cconstruct_model', title='CNN — Model 1 — Simple Architecture'))