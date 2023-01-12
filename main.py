from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

import numpy as np
from sklearn.pipeline import Pipeline

kategoriler = ['software', "humanresource", "endustrimuhendisi"]

hrms = load_files("hrms/jobadvertisements/",
                          categories=kategoriler,
                          shuffle=True,
                          random_state=42,
                          encoding='utf-8',
                          decode_error='ignore')

print(type(hrms))

print(hrms.target_names)

print(len(hrms.data))

print(hrms.target[:5])

count_vect = CountVectorizer()

# #fit_transform fonksiyonuyla egitim verisindeki oznitelikleri secer
X_train_counts = count_vect.fit_transform(hrms.data)

print(X_train_counts.shape)


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)

X_train_tf = tf_transformer.transform(X_train_counts)

print(X_train_tf.shape)

clf = MultinomialNB().fit(X_train_tf, hrms.target)

docs_news = ['C, C++, C#, Java, Python, SQL, JavaScript, HTML, CSS, PHP, ASP.Net Core, Node.js, Microsoft Office Programları, React.js, Bootstrap, Entity Framework, Express.js, Angular, MongoDb, Asp.Net Core Web Api, Asp.Net Core Mvc', "ERP", "ücretlendirme, performans yönetimi", "MS Office"]

X_new_count = count_vect.transform(docs_news)
print(X_new_count.shape)

X_new_tf = tf_transformer.transform(X_new_count)
print(X_new_tf.shape)

predicted = clf.predict(X_new_tf)

text_clf1 = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB())])

text_clf1.fit(hrms.data, hrms.target)

hrms_test = load_files("hrms/jobadvertisements-test/",
                         categories=kategoriler,
                         shuffle=True,
                         random_state=42,
                         encoding='utf-8',
                         decode_error='ignore')

docs_test = hrms_test.data

predicted2 = text_clf1.predict(docs_test)

print(np.mean(predicted2 == hrms_test.target))

#tahmin edilen sınıfları ekrana yazdıralım
for doc, category in zip(docs_news, predicted):
    print('%r=>%s' % (doc, hrms.target_names[category]))