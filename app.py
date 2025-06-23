import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By 
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from transformers import pipeline
import pandas as pd
import plotly.express as px

def get_youtube_comments(video_url, max_comments=1000):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver = webdriver.Chrome(
    service=Service(ChromeDriverManager(driver_version="137.0.7151.120").install()),
    options=options)

    st.info("🚀 Opening video...")
    driver.get(video_url)
    time.sleep(5)

    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height or len(driver.find_elements(By.XPATH, '//*[@id="content-text"]')) >= max_comments:
            break
        last_height = new_height

    comments_elements = driver.find_elements(By.XPATH, '//*[@id="content-text"]')
    comments = [elem.text for elem in comments_elements[:max_comments]]

    driver.quit()
    return comments

# 💅 Page Config
st.set_page_config(page_title="YouTube Sentiment Analysis", layout="wide", page_icon="📺")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #111;
            color: #eee;
        }
        h1, h2, h3, h4 {
            text-align: center;
            color: #F72585;
        }
        .stButton>button {
            background-color: #7209b7;
            color: white;
            font-size: 16px;
            padding: 5px 24px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📺 YouTube Comments Sentiment Analysis")
st.markdown("## 🎯 Extract YouTube comments, analyze sentiment, and visualize results beautifully!")

# Input Layout
col1, col2 = st.columns([2, 2])
with col1:
    video_url = st.text_input("🔗 Enter YouTube Video URL:")

with col2:
    max_comments = st.slider("💬 Max Comments", min_value=10, max_value=1000, step=10, value=200)

if st.button("🔍 Extract and Analyze"):
    if not video_url:
        st.error("🚨 Please enter a valid YouTube video URL.")
    else:
        with st.spinner("🕵️ Extracting comments..."):
            comments = get_youtube_comments(video_url, max_comments)
        
        if comments:
            st.success(f"✅ Extracted {len(comments)} comments.")

            with st.expander("📜 Show Extracted Comments"):
                for c in comments[:50]:
                    st.write(f"👉 {c}")

            # Sentiment Analysis
            st.info("🔬 Analyzing sentiment...")
            progress = st.progress(0)
            pipe = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
            result = []
            for idx, comment in enumerate(comments):
                res = pipe(comment, truncation=True)
                result.append(res[0])
                progress.progress((idx+1)/len(comments))
            progress.empty()

            # DataFrame
            df = pd.DataFrame({
                'Comment': comments,
                'Label': [res['label'] for res in result],
                'Score': [res['score'] for res in result]
            })

            st.dataframe(df.head())

            csv_file = 'youtube_comments_sentiment.csv'
            st.download_button(label="💾 Download CSV", data=df.to_csv(index=False).encode('utf-8'), file_name=csv_file, mime='text/csv')

            # Custom Plotly Colors
            sentiment_counts = df['Label'].value_counts()
            fig = px.pie(
                names=sentiment_counts.index,
                values=sentiment_counts.values,
                color=sentiment_counts.index,
                color_discrete_map={'POSITIVE': '#00FF7F', 'NEGATIVE': '#FF4500'},
                title="🎨 YouTube Comments Sentiment Distribution"
            )
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='black',
                plot_bgcolor='black',
                title_font=dict(size=24, color='#F72585'),
                legend=dict(font=dict(color='white'), bgcolor='black'),
                font=dict(family="Arial", size=14)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("😕 No comments were extracted from the video.")
