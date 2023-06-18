import streamlit as st
import pandas as pd
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import klasifikasi

df = pd.read_csv('data.csv')
df = df.drop('Unnamed: 0.1', axis=1)
df = df.drop('Unnamed: 0', axis=1)

def main():

    # Menambahkan kelas CSS pada container sidebar
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            transition: margin-left 200ms;
        }
        .sidebar:hover .sidebar-content {
            margin-left: 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
    # Membuat container sidebar
    sidebar_container = st.empty()

    # Menampilkan/menyembunyikan sidebar berdasarkan hover
    with sidebar_container:
        option = st.sidebar.selectbox('Menu Layanan', ['Home Page','Internet', 'Telepon','TV'])
        
        
    container = st.container()
    if option != "Internet" and option != "Klasifikasi" and option != "Telepon" and option != "TV":
        container.title('Dashboard Analisis Sentimen Layanan Indihome')
        container.write('<div style="text-align: justify;">Selamat datang di dashboard sederhana ini, dalam analisis sentimen Indihome di Twitter, dengan menggunakan metode K-Nearest Neighbors (KNN) untuk menganalisis sentimen dari tweet-tweet yang terkait dengan layanan Indihome. Metode KNN adalah salah satu metode dalam machine learning yang digunakan untuk klasifikasi data berdasarkan kemiripan dengan data pelatihan yang ada.</div><br>', unsafe_allow_html=True)
        container.write('<div style="text-align: justify;">Sumber data berasal Twitter yang terdiri dari 1000 data terkait dengan layanan Indihome. Data tersebut mencakup beragam tweet yang berhubungan dengan pengalaman pengguna terkait Indihome </div><br>', unsafe_allow_html=True)
        container.write('<div style="text-align: justify;">Labelling awal menggunakan metode <i>Lexicon Based</i>. Lexicon yang di gunakan berasal dari Kamus Inset, yang merupakan sumber referensi yang berasal dari tweet. Dengan menggunakan metode KNN didapat akurasi sebesar 76% dengan nilai k=1</div>', unsafe_allow_html=True)
         # Bagian utama di sebelah kanan dengan tata letak rata kiri dan kanan
   
        
    if option == "Internet":
        st.subheader("Halaman Layanan Internet")
        tab1, tab2 = st.tabs(["Wordcloud", "Data"])
        
        with tab1:
            option_layanan = st.selectbox('Sentimen', ['Positive','Negative'], key='layanan_internet_sentimen')
            
            if option_layanan == 'Positive':
                # Menggabungkan teks dari kolom tweet_bersih berdasarkan sentimen positive
                teks_positive_internet = ' '.join(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('internet'))]['tweet_text_prepocessing_stem'])

                # Membuat WordCloud dari teks positive
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_positive_internet)

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(fig)
            else:
                # Menggabungkan teks dari kolom tweet_bersih berdasarkan sentimen positive
                teks_negatif_internet = ' '.join(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('internet'))]['tweet_text_prepocessing_stem'])

                # Membuat WordCloud dari teks positive
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_negatif_internet)

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(fig)

        with tab2:
            data_positive_internet = df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('internet'))]
            data_negative_internet = df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('internet'))]
           
            data_positive_renamed = data_positive_internet.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})
            data_negative_renamed = data_negative_internet.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})          
           
            jumlah_positive_internet = len(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('internet'))])
            jumlah_negatif_internet = len(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('internet'))])
            
            
            if not data_positive_internet.empty:
                st.subheader("Data Sentimen positive layanan indihome 'internet'")
                st.write( data_positive_renamed[['Tweet', 'Label']])
                st.write("Jumlah sentimen positive pada layanan Internet: ", jumlah_positive_internet, "Data")
                
                st.subheader("Data Sentimen negative layanan indihome 'internet'")
                st.write(data_negative_renamed[['Tweet', 'Label']])
                st.write("Jumlah sentimen negatif pada layanan Internet: ", jumlah_negatif_internet, "Data")
                
    elif option == "Telepon":
        tab1, tab2 = st.tabs(["Wordcloud", "Data"])
        with tab1:
            jumlah_positive_telepon = len(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))])
            jumlah_negatif_telepon = len(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))])

            option_telepon = st.selectbox('Sentimen', ['Positive','Negative'], key='layanan_telepon_sentimen')
           
            if option_telepon == 'Positive':
                # Menggabungkan teks dari kolom tweet_bersih berdasarkan sentimen positive
                teks_positive_telepon = ' '.join(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))]['tweet_text_prepocessing_stem'])

                # Membuat WordCloud dari teks positive
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_positive_telepon)

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(fig)
            else:
                # Menggabungkan teks dari kolom tweet_bersih berdasarkan sentimen positive
                teks_negative_telepon = ' '.join(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))]['tweet_text_prepocessing_stem'])

                # Membuat WordCloud dari teks positive
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_negative_telepon)

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(fig)    
                 
        with tab2:
            data_positive_telepon = df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))]
            data_negative_telepon = df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))]
           
            data_positive_telepon_renamed = data_positive_telepon.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})
            data_negative_telepon_renamed = data_negative_telepon.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})          
           
            jumlah_positive_telepon = len(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))])
            jumlah_negatif_telepon = len(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('telepon'))])
            
            
            if not data_positive_telepon.empty:
                st.subheader("Data Sentimen positive layanan indihome 'internet'")
                st.write( data_positive_telepon_renamed[['Tweet', 'Label']])
                st.write("Jumlah sentimen positive pada layanan Internet: ", jumlah_positive_telepon, "Data")
                
                st.subheader("Data Sentimen negative layanan indihome 'internet'")
                st.write(data_negative_telepon_renamed[['Tweet', 'Label']])
                st.write("Jumlah sentimen negatif pada layanan Internet: ", jumlah_negatif_telepon, "Data")   
                         
    elif option == "TV":
        tab1, tab2 = st.tabs(["Wordcloud", "Data"])
        with tab1:
            jumlah_positive_tv = len(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))])
            jumlah_negatif_tv = len(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))])
            
            
            option_tv = st.selectbox('Sentimen', ['Positive','Negative'], key='layanan_tv_sentimen')
            
            if option_tv == 'Positive':
                # Menggabungkan teks dari kolom tweet_bersih berdasarkan sentimen positive
                teks_positive_tv = ' '.join(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))]['tweet_text_prepocessing_stem'])

                # Membuat WordCloud dari teks positive
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_positive_tv)

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(fig)
            else:
                # Menggabungkan teks dari kolom tweet_bersih berdasarkan sentimen positive
                teks_negative_tv = ' '.join(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))]['tweet_text_prepocessing_stem'])

                # Membuat WordCloud dari teks positive
                wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_negative_tv)

                fig, ax = plt.subplots(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title('Word Cloud')
                st.pyplot(fig)    
                 
        with tab2:
            data_positive_tv = df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))]
            data_negative_tv = df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))]
           
            data_positive_tv_renamed = data_positive_tv.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})
            data_negative_tv_renamed = data_negative_tv.rename(columns={'tweet_text_prepocessing_stem': 'Tweet', 'sentiment_predict': 'Label'})          
           
            jumlah_positive_tv = len(df[(df['sentiment_predict'] == 0) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))])
            jumlah_negatif_tv = len(df[(df['sentiment_predict'] == 1) & (df['tweet_text_prepocessing_stem'].str.contains('tv'))])
            
            
            if not data_positive_tv.empty:
                st.subheader("Data Sentimen positive layanan indihome 'internet'")
                st.write( data_positive_tv_renamed[['Tweet', 'Label']])
                st.write("Jumlah sentimen positive pada layanan Internet: ", jumlah_positive_tv, "Data")
                
                st.subheader("Data Sentimen negative layanan indihome 'internet'")
                st.write(data_negative_tv_renamed[['Tweet', 'Label']])
                st.write("Jumlah sentimen negatif pada layanan Internet: ", jumlah_negatif_tv, "Data")    
                          
if __name__ == '__main__':
    main()