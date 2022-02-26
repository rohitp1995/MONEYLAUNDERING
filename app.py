import streamlit as st
import pandas as pd

#import time

@st.cache
def readcsv(csv):
    df = pd.read_csv(csv)
    return df


def main():

    st.title('Money Laundering Detection App')
    st.image('image.jpg',width=2000, use_column_width='always')
    st.markdown('Upload a proper csv file')
    file = st.file_uploader('Upload your csv file', type='csv')

    if file is not None:
        
        df0 = pd.DataFrame(readcsv(file))
        
if __name__ == '__main__':
    main()