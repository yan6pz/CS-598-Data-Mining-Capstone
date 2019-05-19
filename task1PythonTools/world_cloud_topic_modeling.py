import graphlab as gl
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
topic_size = 10
reviews = gl.SFrame.read_json('yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json',orient='lines')
reviews = reviews[(reviews['stars'] == 1)]
print(reviews.head())

reviews.remove_columns(['business_id','date','review_id','user_id','votes','type'])
stopwords = gl.text_analytics.stopwords() | set(stopwords.words('english'))
delimiters=["\r", "\v", "\n", "\f", "\t", " ",'-','.',',','?','&','!',':',';','"','(',')','[',']','{','}','=','/']


reviews['word_count'] = gl.text_analytics.count_words(reviews['text'],delimiters = delimiters).dict_trim_by_keys(stopwords, exclude=True)
reviews['tf_idf'] = gl.text_analytics.tf_idf(reviews['word_count'])


sample = reviews.sample(.5,seed = 317)
sample.head()


def model_cgs(data):
    return gl.topic_model.create(data,num_topics=topic_size, num_iterations=200,print_interval=50,method='cgs')
def model_alias(data):
    return gl.topic_model.create(data, num_topics=topic_size, num_iterations=50,print_interval=50,method='alias')

# collapsed gibbs sampling
topic_model_cgs = model_cgs(sample['tf_idf'])


def plot_word_cloud(ls):
    d={}
    for a,x in ls:
        d[a]=x

    wordcloud = WordCloud(background_color='white').generate_from_frequencies(d)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()

def print_model(model):
    words = model.get_topics(num_words=100).pack_columns(columns=['word','score'],new_column_name='value')
    #dataframe = words.to_dataframe().set_index('topic')
    print(words)


    #with open('words.txt', 'wb') as file:
    #   file.write(d)

    for i in range(topic_size):
        print 'Topic %d:'%i
        dt= words.filter_by(i,'topic')

        print(dt['value'])
        plot_word_cloud(dt['value'])


print_model(topic_model_cgs)