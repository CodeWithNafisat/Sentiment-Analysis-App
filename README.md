# Sentiment Analysis App

## About This App

This is a sentiment analysis tool that uses machine learning to figure out if movie reviews are positive or negative. It's built to help people quickly understand audience feedback without reading through hundreds of reviews manually.

The app takes the technical work I've done with machine learning models and makes it accessible through a simple, interactive interface that anyone can use.

## What It Does

* **Text Cleaning:** Automatically processes and prepares text for analysis by handling things like tokenization and stopword removal
* **Model Choice:** Lets you pick between two different analysis methods—Logistic Regression or Support Vector Machine (SVM)
* **Multiple Input Options:** You can analyze single reviews or upload a CSV file to process many reviews at once
* **Confidence Scores:** Shows how confident the model is about each prediction so you know how much to trust the results

## The Problem I'm Solving

### The Challenge

These days, movies and shows get thousands of online reviews. Reading through all of them to understand overall audience sentiment takes forever and can be subjective. While automated sentiment analysis exists, making it actually usable for real people remains difficult.

### My Solution

I built this app to:

* Automatically classify review sentiment using natural language processing
* Provide a simple web interface for instant analysis
* Let users compare how different models perform
* Show how trained models can be turned into practical tools people can actually use


## How to Get Started

1. **Get the Code**

   ```bash
   git clone https://github.com/CodeWithNafisat/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```

2. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Check for Model Files**
   Make sure these files are in your folder:

   * `regression_sentiment.pkl`
   * `svm_sentiment.pkl`
   * `vectorizer.pkl`
   * `svd.pkl`

4. **Run the App**

   ```bash
   streamlit run app.py
   ```

5. **Open the App**
   Once it's running, go to `http://localhost:8501/` in your browser to start using it.

---

## How to Use It

* Type or paste a movie review into the text box
* Choose which model you want to use: Logistic Regression or SVM
* Click "Analyze" to see the sentiment result and confidence score
* For analyzing multiple reviews, upload a CSV file with a column called **Review**


## Technical Stuff

* **Python** – handles the backend logic and model integration
* **Streamlit** – creates the web interface
* **Scikit-learn** – used for model training and evaluation
* **Joblib** – manages model saving and loading
* **NLP preprocessing** – cleans and prepares text for analysis

---

## Project Layout

```
sentiment-analysis-app/
│
├─ app.py
├── requirements.txt
├── regression_sentiment.pkl
├── svm_sentiment.pkl
├── vectorizer.pkl
├── svd.pkl
├── data/
└── README.md
```

---

## About Me

**Nafisat Abdulraheem**
I'm passionate about data science and machine learning, and I focus on building AI tools that are actually useful and understandable for real people solving real problems.

* **GitHub:** [CodeWithNafisat](https://github.com/CodeWithNafisat)
* **LinkedIn:** [Nafisat Abdulraheem](https://www.linkedin.com/in/nafisat-abdulraheem-7a193b337)

---

## License

This project uses the MIT License. Check the LICENSE file for details.
