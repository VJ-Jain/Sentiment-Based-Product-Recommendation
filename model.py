
import pickle
import numpy as np
import pandas as pd

def get_product_recommendations(user_name, count = 5):
  # Load exported model files
  tfidf = pickle.load(open('tfidf.pkl', 'rb'))
  df_prep = pickle.load(open('df_prep.pkl', 'rb'))
  user_final_rating = pickle.load(open('user_final_rating.pkl', 'rb'))
  final_sentiment_model = pickle.load(open('final_sentiment_model.pkl', 'rb'))

  if count >20:
    return ("null", "Error")

  if user_name not in user_final_rating.index:
    return ("null", "Error")
  
  # Get a list of top recommended products
  recommendations = list(user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index)
  # Find all the reviews of top recommended products
  recommendation_reviews = df_prep[df_prep.id.isin(recommendations)]
  # Predict sentiments of reviews
  recommendation_withPredictedSentiments = recommendation_reviews.copy()
  X = tfidf.transform(recommendation_reviews["reviews"].values.astype(str))
  recommendation_withPredictedSentiments['predicted_sentiment'] = final_sentiment_model.predict(X)
  # Calculate number of total reviews for each recommended product
  recommendation_withPredictedSentiments = recommendation_withPredictedSentiments[['name', 'predicted_sentiment']]
  recommendations_grouped = recommendation_withPredictedSentiments.groupby('name', as_index=False).count()
  # Find number of positive review sentiments for each product
  recommendations_grouped["positive_review_count"] = recommendations_grouped.name.apply(lambda x: recommendation_withPredictedSentiments[(recommendation_withPredictedSentiments.name==x) & (recommendation_withPredictedSentiments.predicted_sentiment==1)]["predicted_sentiment"].count())
  recommendations_grouped["total_review_count"] = recommendations_grouped['predicted_sentiment']
  # Find percentage of positive review sentiments for each product
  recommendations_grouped['positive_sentiment_percent'] = np.round(recommendations_grouped["positive_review_count"]/recommendations_grouped["total_review_count"]*100,2)
  # Find Top 5 products with highest percentage of positive reviews
  recommendations_grouped = recommendations_grouped.sort_values('positive_sentiment_percent', ascending=False)

  return (recommendations_grouped[['name']].head(count).to_html(), "No Error")