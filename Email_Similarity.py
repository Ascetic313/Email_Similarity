from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

emails = fetch_20newsgroups()
all_categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
predictions = []
categories =[]
for i in range(len(all_categories)):
  for j in range(len(all_categories)):
    categories.append(all_categories[i])
    categories.append(all_categories[j])
    train_emails = fetch_20newsgroups(categories = categories, subset='train', shuffle = True, random_state = 108)
    test_emails = fetch_20newsgroups(categories = categories, subset='test', shuffle = True, random_state = 108)

    counter = CountVectorizer()
    counter.fit(test_emails.data + train_emails.data)
    train_counts = counter.transform(train_emails.data)
    test_counts = counter.transform(test_emails.data)

    classifier = MultinomialNB()
    classifier.fit(train_counts, train_emails.target)
    predictions.append(categories[0] + " " + categories[1] + " " + str(classifier.score(test_counts, test_emails.target)))

print(predictions)