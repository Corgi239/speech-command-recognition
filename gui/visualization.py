import streamlit as st
import tensorflow as tf
import visualkeras

MODEL_PATH = 'model_CNN/model.h5'
SAMPLES_TO_CONSIDER = 22050


mapping=[
        "right",
        "eight",
        "cat",
        "tree",
        "backward",
        "learn",
        "bed",
        "happy",
        "go",
        "dog",
        "no",
        "wow",
        "follow",
        "nine",
        "left",
        "stop",
        "three",
        "sheila",
        "one",
        "bird",
        "zero",
        "seven",
        "up",
        "visual",
        "marvin",
        "two",
        "house",
        "down",
        "six",
        "yes",
        "on",
        "five",
        "forward",
        "off",
        "four"
        ]


# @st.cache
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    """Extract MFCCs from audio file.

    :param file_path (str): Path of audio file
    :param num_mfcc (int): # of coefficients to extract
    :param n_fft (int): Interval we consider to apply STFT. Measured in # of samples
    :param hop_length (int): Sliding window for STFT. Measured in # of samples
    :return MFCCs (ndarray): 2-dim array with MFCC data of shape (# time steps, # coefficients)
    """

    # load audio file
    signal, sample_rate = librosa.load(file_path)

    if len(signal) >= SAMPLES_TO_CONSIDER:
        # ensure consistency of the length of the signal
        signal = signal[:SAMPLES_TO_CONSIDER]

        # extract MFCCs
        MFCCs = librosa.feature.mfcc(signal, sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                         hop_length=hop_length)
        return MFCCs.T




def CNN_predict(file_path):
    # TODO: implement a function that recognizes the recording in the file using the tensorflow model
    """
    :param file_path (str): Path to audio file to predict
    :return predicted_keyword (str): Keyword predicted by the model
    """

    # extract MFCC
    MFCCs = preprocess(file_path)

    # we need a 4-dim array to feed to the model for prediction: (# samples, # time steps, # coefficients, 1)
    MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

    # get the predicted label
    predictions = model.predict(MFCCs)
    predicted_index = np.argmax(predictions)
    predicted_keyword = mapping[predicted_index]
    return predicted_keyword


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
