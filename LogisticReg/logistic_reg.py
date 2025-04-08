import numpy as np
import pandas as pd
import re 
import string
import nltk

nltk.download('punkt')  # This downloads necessary resources.
from nltk.tokenize import word_tokenize

# stopwords: common words that might be less informative.
from nltk.corpus import stopwords
nltk.download('stopwords')

# to convert words to their base form (lemmatization)
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')


class LogisticReg:

    def __init__(self):
        pass

    # Added self parameter and fixed the sigmoid function.
    def sigmoid(self, x):
        # Clip x to avoid overflow. Corrected clip usage: np.clip(x, -250, 250)
        # Also fixed the division so that the entire denominator is computed.
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    # High-level function: cleans the email data for better computing.
    def preproces(self,email):
        email = email.lower()
        email = email.join(text.split())

        # Remove email addresses
        emailpattern = r'\S+@\S+'
        email = re.sub(emailpattern, '', email)

        # Remove URLs
        urlpattern = r'http[s]?://\S+'
        email = re.sub(urlpattern, '', email)

        # Remove punctuation
        punctuation_pattern = "[" + re.escape(string.punctuation) + "]"
        email = re.sub(punctuation_pattern, '', email)

        # Tokenization 
        tokens = word_tokenize(email)  
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        clean_email = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
        return clean_email

    def vocabulary(self, emails):
        words_set = set()
        for email in emails:
            for word in email:
                # Use the local set variable, not the built-in set.
                words_set.add(word)
        # Return as a sorted list for consistent ordering.
        return sorted(list(words_set))
    
    def vectorization(self, vocabulary, words):
        # Create a binary feature vector: 1 if word present, 0 otherwise.
        feature_vec = [0] * len(vocabulary)
        for i, word in enumerate(vocabulary):
            if word in words:
                feature_vec[i] = 1
        return feature_vec

    def loglikelihood(self,y_predicted,Y):
        # Calculate cross-entropy log likelihood.
        return Y * np.log(y_predicted + 1e-8) + (1 - Y) * np.log(1 - y_predicted + 1e-8)
    
    def train(self, emails, Y,epochs,lr):
        # Preprocess emails
        cleaned_emails=[]
        for email in emails:
            cleaned_emails.append(self.prepcoess(email))
        
        emails = cleaned_emails

        # Create vocabulary from processed emails
        vocabulary = self.vocabulary(emails)

        # Vectorization of emails: create a 2D array (each row is a feature vector)
        emails_vec = []
        for email in emails:
            vec_email = self.vectorization(vocabulary, email)
            emails_vec.append(vec_email)
        emails_vec = np.array(emails_vec)  # Convert to numpy array


        m=Y.shape[0] 
        
        X_aug = np.concatenate((emails_vec, np.ones((m, 1))), axis=1)
        
        theta = np.zeros(X_aug.shape[1])

        for epoch in range(epochs):

            z = X_aug.dot(theta)
            y_predicted = self.sigmoid(z)
            error = y_predicted - Y  # shape: (m,)

            #diagonal matrix W where W[i, i] = y_predicted[i] * (1 - y_predicted[i])
            W = np.diag(y_predicted * (1 - y_predicted))
            
            H = X_aug.T.dot(W).dot(X_aug) 
            grad = X_aug.T.dot(error)

            # Newton update
            theta = theta - np.linalg.inv(H).dot(grad)

            loss = - (1/m) * np.sum(self.loglikelihood(y_predicted,Y))
            print(f"Epoch {epoch}, Loss: {loss}")

        
        beta = theta[:-1]
        b = theta[-1]
        
        return beta, b
    
    def predict(self, emails, beta, b, vocabulary):
        clean_emails = [self.preproces(email) for email in emails]
        
        # Vectorize each processed email using the provided vocabulary.
        emails_vec = []
        for email in clean_emails:
            vec_email = self.vectorization(vocabulary, email)
            emails_vec.append(vec_email)

        emails_vec = np.array(emails_vec)  # Shape: (num_emails, num_features)
        
        #z = X * beta + b.
        z = emails_vec.dot(beta) + b
        
        # predicted probabilites
        y_preds = self.sigmoid(z)
        
        return y_preds


# TODO : Implement croos validation similar to this from Ridge Reg    
    # def bestbeta(X,Y,k,lambdas,lr,epochs):
    #     if Y.ndim==1:
    #         Y.reshape(-1,1)
        
    #     m=X.shape[0]
        
    #     #Random permuatations or indexed X,Y for better results
    #     indices = np.random.permutation(m)
    #     X_shuf=X[indices]
    #     Y_shuf= Y[indices]

    #     #Split data into k folds
    #     X_folds = np.array_split(X_shuf,k)
    #     Y_folds = np.array_split(Y_shuf,k)

    #     results = []
        
    #     for lambda_ in lambdas:
    #         mse_list=[]

    #         for i in range(k):
    #             #test or validation folds
    #             val_x = X_folds[i]
    #             val_y=Y_folds[i]

    #             #training folds
    #             train_ind = [j for j in range(k) if i!=j]
    #             train_x=np.vstack([X_folds[j]  for j in train_ind])
    #             train_y=np.vstack([Y_folds[j]  for j in train_ind])


    #             theta = np.random.randn(X.shape[1],1)


    #             pred_Y = val_x @ theta
    #             mse = np.mean((pred_Y - val_y) ** 2)
    #             mse_list.append(mse)

    #         #mean avg error 
    #         avg_mse = np.mean(mse_list)
    #         results.append((lambda_, avg_mse))
    #     results.sort(key=lambda x: x[1] )

    #     print(results)
    #     return results[0][0]