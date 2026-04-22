import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import os

class CommercialHybridEngine:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.svd = SVD(n_factors=100, n_epochs=20)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.movies_df = None
        self.content_sim_matrix = None

    def load_and_train(self):
        print("[Step 1] Loading Dataset")
        data = Dataset.load_builtin('ml-100k')
        trainset = data.build_full_trainset()
        
        print("[Step 2] Training Collaborative Filter (SVD)...")
        self.svd.fit(trainset)

        print("[Step 3] Processing Movie Metadata for Content-Based Filter...")
        
        url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
        columns = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 
                   'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                   'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        
        self.movies_df = pd.read_csv(url, sep='|', names=columns, encoding='latin-1')
        
        genre_cols = columns[6:]
        self.movies_df['metadata'] = self.movies_df[genre_cols].apply(
            lambda x: ' '.join(x.index[x==1]), axis=1
        )

        tfidf_matrix = self.tfidf.fit_transform(self.movies_df['metadata'])
        self.content_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("[Success] Model is trained and ready for production.")

    def recommend(self, user_id, top_n=5):
        all_movie_ids = self.movies_df['movie_id'].unique()
        
        predictions = []
        for mid in all_movie_ids:
            cf_score = self.svd.predict(str(user_id), str(mid)).est
            
            content_score = 3.0 
            
            final_score = (self.alpha * cf_score) + ((1 - self.alpha) * content_score)
            predictions.append((mid, final_score))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for mid, score in predictions[:top_n]:
            title = self.movies_df[self.movies_df['movie_id'] == mid]['title'].values[0]
            results.append({'title': title, 'score': round(score, 2)})
        
        return results

if __name__ == "__main__":
    engine = CommercialHybridEngine(alpha=0.8)
    engine.load_and_train()
    
    recs = engine.recommend(user_id=420)
    
    print("\n--- Top 5 Recommendations for User 420 ---")
    for r in recs:
        print(f"{r['title']} (Predicted Rating: {r['score']})")