import numpy as np
import pandas as pd
import pickle
import time

import warnings 
warnings.filterwarnings("ignore")

import maha

husna = pd.read_csv("husna.csv")

# Make Copy
husna_copy = husna.copy()

# Delete Null Contents
husna_copy = husna_copy[husna_copy['content'] != '[]']

# Clean the content by making it a str not a list
husna_copy['new_content'] = husna['content'].apply(lambda x: x[2:-2])

# Clean tags
def clean_tags(tag):
    lst = [i.replace("'","") for i in tag.split(',')]
    lst[0] = lst[0][1:]
    lst[-1] = lst[-1][:-1]
    return lst

husna_copy['new_tags'] = husna_copy['tags'].apply(clean_tags)
#husna_copy['new_tags'] = husna_copy['new_tags'].apply()

content = list(husna_copy['new_content'])
husna_copy.head(2)

# Stops Words ---------------------------------------
import nltk
nltk.download('stopwords')
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords

stops = set(stopwords.words('arabic'))
stop_word_comp = {"،","%","آض","آمينَ","آه","آهاً","آي","أ","أب","أجل","أجمع","أخ","أخذ","أصبح","أضحى","أقبل","أقل","أكثر","ألا","أم","أما","أمامك","أمامكَ","أمسى","أمّا","أن","أنا","أنت","أنتم","أنتما","أنتن","أنتِ","أنشأ","أنّى","أو","أوشك","أولئك","أولئكم","أولاء","أولالك","أوّهْ","أي","أيا","أين","أينما","أيّ","أَنَّ","أََيُّ","أُفٍّ","إذ","إذا","إذاً","إذما","إذن","إلى","إليكم","إليكما","إليكنّ","إليكَ","إلَيْكَ","إلّا","إمّا","إن","إنّما","إي","إياك","إياكم","إياكما","إياكن","إيانا","إياه","إياها","إياهم","إياهما","إياهن","إياي","إيهٍ","إِنَّ","ا","ابتدأ","اثر","اجل","احد","اخرى","اخلولق","اذا","اربعة","ارتدّ","استحال","اطار","اعادة","اعلنت","اف","اكثر","اكد","الألاء","الألى","الا","الاخيرة","الان","الاول","الاولى","التى","التي","الثاني","الثانية","الذاتي","الذى","الذي","الذين","السابق","الف","اللائي","اللاتي","اللتان","اللتيا","اللتين","اللذان","اللذين","اللواتي","الماضي","المقبل","الوقت","الى","اليوم","اما","امام","امس","ان","انبرى","انقلب","انه","انها","او","اول","اي","ايار","ايام","ايضا","ب","بات","باسم","بان","بخٍ","برس","بسبب","بسّ","بشكل","بضع","بطآن","بعد","بعض","بك","بكم","بكما","بكن","بل","بلى","بما","بماذا","بمن","بن","بنا","به","بها","بي","بيد","بين","بَسْ","بَلْهَ","بِئْسَ","تانِ","تانِك","تبدّل","تجاه","تحوّل","تلقاء","تلك","تلكم","تلكما","تم","تينك","تَيْنِ","تِه","تِي","ثلاثة","ثم","ثمّ","ثمّة","ثُمَّ","جعل","جلل","جميع","جير","حار","حاشا","حاليا","حاي","حتى","حرى","حسب","حم","حوالى","حول","حيث","حيثما","حين","حيَّ","حَبَّذَا","حَتَّى","حَذارِ","خلا","خلال","دون","دونك","ذا","ذات","ذاك","ذانك","ذانِ","ذلك","ذلكم","ذلكما","ذلكن","ذو","ذوا","ذواتا","ذواتي","ذيت","ذينك","ذَيْنِ","ذِه","ذِي","راح","رجع","رويدك","ريث","رُبَّ","زيارة","سبحان","سرعان","سنة","سنوات","سوف","سوى","سَاءَ","سَاءَمَا","شبه","شخصا","شرع","شَتَّانَ","صار","صباح","صفر","صهٍ","صهْ","ضد","ضمن","طاق","طالما","طفق","طَق","ظلّ","عاد","عام","عاما","عامة","عدا","عدة","عدد","عدم","عسى","عشر","عشرة","علق","على","عليك","عليه","عليها","علًّ","عن","عند","عندما","عوض","عين","عَدَسْ","عَمَّا","غدا","غير","ـ","ف","فان","فلان","فو","فى","في","فيم","فيما","فيه","فيها","قال","قام","قبل","قد","قطّ","قلما","قوة","كأنّما","كأين","كأيّ","كأيّن","كاد","كان","كانت","كذا","كذلك","كرب","كل","كلا","كلاهما","كلتا","كلم","كليكما","كليهما","كلّما","كلَّا","كم","كما","كي","كيت","كيف","كيفما","كَأَنَّ","كِخ","لئن","لا","لات","لاسيما","لدن","لدى","لعمر","لقاء","لك","لكم","لكما","لكن","لكنَّما","لكي","لكيلا","للامم","لم","لما","لمّا","لن","لنا","له","لها","لو","لوكالة","لولا","لوما","لي","لَسْتَ","لَسْتُ","لَسْتُم","لَسْتُمَا","لَسْتُنَّ","لَسْتِ","لَسْنَ","لَعَلَّ","لَكِنَّ","لَيْتَ","لَيْسَ","لَيْسَا","لَيْسَتَا","لَيْسَتْ","لَيْسُوا","لَِسْنَا","ما","ماانفك","مابرح","مادام","ماذا","مازال","مافتئ","مايو","متى","مثل","مذ","مساء","مع","معاذ","مقابل","مكانكم","مكانكما","مكانكنّ","مكانَك","مليار","مليون","مما","ممن","من","منذ","منها","مه","مهما","مَنْ","مِن","نحن","نحو","نعم","نفس","نفسه","نهاية","نَخْ","نِعِمّا","نِعْمَ","ها","هاؤم","هاكَ","هاهنا","هبّ","هذا","هذه","هكذا","هل","هلمَّ","هلّا","هم","هما","هن","هنا","هناك","هنالك","هو","هي","هيا","هيت","هيّا","هَؤلاء","هَاتانِ","هَاتَيْنِ","هَاتِه","هَاتِي","هَجْ","هَذا","هَذانِ","هَذَيْنِ","هَذِه","هَذِي","هَيْهَاتَ","و","و6","وا","واحد","واضاف","واضافت","واكد","وان","واهاً","واوضح","وراءَك","وفي","وقال","وقالت","وقد","وقف","وكان","وكانت","ولا","ولم","ومن","مَن","وهو","وهي","ويكأنّ","وَيْ","وُشْكَانََ","يكون","يمكن","يوم","ّأيّان"}

# Union of the two sets
full_stops = stops.union(stop_word_comp)


def remove_stop_words(text):
    words = text.split()
    return " ".join([w for w in words if w not in full_stops and len(w) >= 2])



# Removing Others ---------------------------------------

from maha.cleaners.functions import remove

def removing(text):
    
    text = remove(text,
           english=True, # English words
           numbers=True, 
           harakat=True, 
           tatweel=True, # Tatweel "-"
           all_harakat=True,
           punctuations=True,
           arabic_punctuations=True,
           english_punctuations=True,
           emails=True, # "@"
           emojis=True, 
           links=True, # "www......"
           mentions=True) # "#" 
    
    return text


# In[312]:


# Normalize ---------------------------------------

from maha.cleaners.functions import normalize
from maha.cleaners.functions import normalize_lam_alef
from maha.cleaners.functions import normalize_small_alef

def normalizing(text):
    text = normalize(text, all=True) # Make All the normalization proccesses
    text = normalize_lam_alef(text) # ت ى --> تا
#    text = normalize_small_alef(text) #  عمّ -->  عمى
    return text



# Stemming ---------------------------------------
from tashaphyne.stemming import ArabicLightStemmer

ArListem = ArabicLightStemmer()

def stemming(text):
    words = text.split()
    cleaned = list()
    for w in words:
        ArListem.light_stem(w)
        cleaned.append(ArListem.get_root())
        
    return " ".join(cleaned)


# In[314]:


# Full PreProcessing
def processing(text):
    
    text = remove_stop_words(text)
    text = removing(text)
    text = normalizing(text)
    text = stemming(text)
    
    return text



with open('cleaned_content.txt', 'rb') as file:
    clean_content = pickle.load(file)
    
sum1, sum2 = 0, 0

for sample in content:
    sum1 += len(sample.split(" "))

for sample in clean_content:
    sum2 += len(sample.split(" "))
    
print("Before: ", sum1)
print("After: ", sum2)
print("Deleted: ", sum1 - sum2)


# In[319]:


from collections import Counter

class Tokenizer:
    def __init__(self, corpus: list[str], min_frequency: int = None):
        self.min_frequency = min_frequency
        self.vocab = self._create_vocab(corpus=corpus)
        
        
    def _create_vocab(self, corpus: list[str]) -> dict[str, int]:
        ...
    
    def _tokenize_document(self, document: str) -> list[int]:
        ...
    
    def tokenize(self, documents: list[str]) -> list[list[int]]:
        return [self._tokenize_document(document) for document in documents]
    
    def __len__(self):
        return self.vocab

class WordLevelTokenizer(Tokenizer):
    def __init__(self, corpus: list[str], min_frequency: int = 0):
        super().__init__(corpus=corpus, min_frequency=min_frequency)
        
    def _create_vocab(self, corpus: list[str]) -> dict[str, int]:
        tokens_counter = Counter([token for sample in corpus for token in sample.split(" ")])
        tokens = [token for token, count in tokens_counter.items() if count >= self.min_frequency]
        vocab = {token: index for index, token in enumerate(tokens, start=2)} 
        vocab["[PAD]"] = 0
        vocab["[OOV]"] = 1
        return vocab
    
    def _tokenize_document(self, document: str) -> list[int]:
        return [self.vocab.get(token, -1) for token in document.split(" ")]

word_level_tokenizer = WordLevelTokenizer(corpus=clean_content, min_frequency=5)
vocab = word_level_tokenizer.vocab

from sklearn.feature_extraction.text import TfidfVectorizer

tf = TfidfVectorizer(vocabulary=vocab)
tf_idf = tf.fit_transform(clean_content)


# In[321]:


def process_query(query):
    
    query = processing(query)
    query = tf.transform([query])
    
    return query


# In[331]:


from sklearn.metrics.pairwise import linear_kernel

def output(query):
    
    query = process_query(query)
    
    cosine_similarities = linear_kernel(query, tf_idf).flatten()
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]
    
    return related_docs_indices, cosine_similarities

# In[332]:

import streamlit as st

# Header
st.markdown("<h1 align = center> Husna Search </h1>", unsafe_allow_html = True)

# Search bar
inp = st.text_input("", placeholder="What do you think about...")

st.markdown("<br>",unsafe_allow_html = True)

col1, col2 = st.columns([1,0.5])

if inp:
    begin = time.time()
    related, cosine = output(inp)
    
    with col1:
        
        st.write(f"[{husna_copy['title'].iloc[related[0]]}]({husna_copy['url'].iloc[related[0]]})")
        st.write(f"[{husna_copy['title'].iloc[related[1]]}]({husna_copy['url'].iloc[related[1]]})")
        st.write(f"[{husna_copy['title'].iloc[related[2]]}]({husna_copy['url'].iloc[related[2]]})")
        st.write(f"[{husna_copy['title'].iloc[related[3]]}]({husna_copy['url'].iloc[related[3]]})")
        st.write(f"[{husna_copy['title'].iloc[related[4]]}]({husna_copy['url'].iloc[related[4]]})")
        
    with col2:
        
        lst = husna_copy['published_at'].iloc[related[0]].split()[0].split('-')
        date = lst[0] + '/' + lst[1] + '/' + lst[2]
        st.markdown(f"**published at:** {date}")
        
        lst = husna_copy['published_at'].iloc[related[1]].split()[0].split('-')
        date = lst[0] + '/' + lst[1] + '/' + lst[2]
        st.markdown(f"**published at:** {date}")
        
        lst = husna_copy['published_at'].iloc[related[2]].split()[0].split('-')
        date = lst[0] + '/' + lst[1] + '/' + lst[2]
        st.markdown(f"**published at:** {date}")
        
        lst = husna_copy['published_at'].iloc[related[3]].split()[0].split('-')
        date = lst[0] + '/' + lst[1] + '/' + lst[2]
        st.markdown(f"**published at:** {date}")
        
        lst = husna_copy['published_at'].iloc[related[4]].split()[0].split('-')
        date = lst[0] + '/' + lst[1] + '/' + lst[2]
        st.markdown(f"**published at:** {date}")
    
    end = time.time()
    time_spent = (end - begin)
    st.markdown(f"<hr> <h5 align = 'center'> search time: {round(time_spent, 3)} second || {round(time_spent,3) * 1000} ms </h5>",unsafe_allow_html = True)

