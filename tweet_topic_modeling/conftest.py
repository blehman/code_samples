import pytest 
import joblib
from gensim.models import LdaModel

@pytest.fixture
def names(scope='session', autouse=True):
    return ["Adila", "Patrick", "Sesh"]

@pytest.fixture
def single_tweet(scope='session', autouse=True):
    return "I have a half pitbull half golden retriever and it's adorable."

@pytest.fixture
def tweet_texts(scope='session', autouse=True):
    return [
        "I have a half pitbull and it's adorable.",
        "I love my golden retriever."
    ]
    
@pytest.fixture
def lda_model(scope='session', autouse=True):
    return LdaModel.load('data/lda/lda_m2symmetric9_20240317-132003.model')

@pytest.fixture
def unique_tweet_texts(scope='session', autouse=True):
    return ['pitbull golden retriever running', 'pitbull golden retriever dancing']    

@pytest.fixture
def tweet_texts_small(scope='session', autouse=True):
    return ['RT @mention half pitbull half golden retriever running', '@mention half pitbull half golden retriever running', 'RT @mention half pitbull half golden retriever dancing', '@mention half pitbull half golden retriever dancing']


@pytest.fixture
def tweet_texts_large(scope='session', autouse=True):
    return ['David u remind me of a golden retriever but a older one &amp; I hope u doing well.',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'Half pitbull half golden retriever #cutepics #aww https://t.co/Bupeqal2H6',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @Goals4Dudes: Half pitbull half golden retriever https://t.co/MNf647Tr6r',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @kayipkopek: #Mersin #Yenişehir 5 aylık Dişi Golden Retriever ŞİLA Kayıp! @mearl1n 05071155930  https://t.co/yPnclH3ETb https://t.co/max…',
 'RT @RetrieverPics: half pitbull half golden retriever https://t.co/q3wkO6mbtZ',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @RetrieverPics: Golden retriever puppies 😍😩 https://t.co/qbc9qhAQcO',
 'RT @Fascinatingpics: This Golden Retriever Snuggling With His Bird And Hamster \n\n7 pics here: https://t.co/teRbt9X5yE\n\n. https://t.co/0gYJ1…',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @kayipkopek: #İzmir #Seferihisar Erkek Golden Retriever Paşa Kayıp! İletişim: 05305113868 @esinvardarcik https://t.co/LdFIkYhdnc https:/…',
 'RT @Goals4Dudes: Half pitbull half golden retriever https://t.co/MNf647Tr6r',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @ashleekmcd: Some days I wish I was a golden retriever in a white upper class family',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR',
 'RT @BabyAnimalPics: half pitbull half golden retriever https://t.co/86vpNGZ9mr',
 'RT @BabyANlMALS_: golden retriever/husky mix https://t.co/Ucd4iCnhmJ',
 'RT @PitbuIIs: half pitbull half golden retriever https://t.co/5Q8blK2MHR']

@pytest.fixture
def train_tokens(scope='session', autouse=True):
    return joblib.load('data/lda/train_tokens_m2_20240317-132003.joblib')

@pytest.fixture
def dictionary(scope='session', autouse=True):
    return joblib.load('data/lda/dictionary_m2_20240317-132003.joblib')

