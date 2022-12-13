import streamlit as st
from PIL import Image
from model_movie import predict
import cv2
import time

movie_path = '../movie/'

def main():
    st.set_option("deprecation.showfileUploaderEncoding", False)

    st.subheader("画像判定アプリ")
    st.write("resnetベース")

    movie_file = st.file_uploader("動画を選択", type=["mp4", "mpeg"])
    image_container = st.empty()
    subheader_st = st.empty()
    labels_st = st.empty()

    if movie_file is not None:
        mv_file_path = movie_path + movie_file.name
        cap_file = cv2.VideoCapture(mv_file_path)
        while cap_file.isOpened():
            #time.sleep(1)
            success, image = cap_file.read()
            if not success:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_container.image(image)
            results = predict(image)
            subheader_st.subheader("判定結果")
            if "safety" == results[0][0]:
                    labels_st.write('<h3><span style="color:blue;">safety</span></h2>', unsafe_allow_html=True)
            elif "warning"  == results[0][0]:
                    labels_st.write('<h3><span style="color: red;">warning</span></h2>', unsafe_allow_html=True)
            else:
                    labels_st.write('<h3><span style="color: gray;">undecidable</span></h2>', unsafe_allow_html=True)

            if cv2.waitKey(5) & 0xFF ==27:
                break
        cap_file.release()

if __name__ == "__main__":
    main()

# streamlit run --server.address localhost app_movie.py
