## Hey YOU!
**Yes, you. Don't try to use the code examples in this README. Instead, download the .ipynb file provided in this repository, fire up [iPython Notebook](http://ipython.org/notebook.html), and run the code there instead. Trust us, you'll like it much better.**

**You can also view a non-runnable version of the notebook (with proper syntax highlighting and embedded images) here: [http://nbviewer.ipython.org/github/charlieg/A-Smattering-of-NLP-in-Python/blob/master/A%20Smattering%20of%20NLP%20in%20Python.ipynb](http://nbviewer.ipython.org/github/charlieg/A-Smattering-of-NLP-in-Python/blob/master/A%20Smattering%20of%20NLP%20in%20Python.ipynb)**

***

# A Smattering of NLP in Python
*by Charlie Greenbacker [@greenbacker](https://twitter.com/greenbacker)*

### Part of a [joint meetup on NLP](http://www.meetup.com/stats-prog-dc/events/177772322/) - 9 July 2014
- [Statistical Programming DC](http://www.meetup.com/stats-prog-dc/)
- [Data Wranglers DC](http://www.meetup.com/Data-Wranglers-DC/)
- [DC Natural Language Processing](http://dcnlp.org/)

***

## Introduction
Back in the dark ages of data science, each group or individual working in
Natural Language Processing (NLP) generally maintained an assortment of homebrew
utility programs designed to handle many of the common tasks involved with NLP.
Despite everyone's best intentions, most of this code was lousy, brittle, and
poorly documented -- not a good foundation upon which to build your masterpiece.
Fortunately, over the past decade, mainstream open source software libraries
like the [Natural Language Toolkit for Python (NLTK)](http://www.nltk.org/) have
emerged to offer a collection of high-quality reusable NLP functionality. These
libraries allow researchers and developers to spend more time focusing on the
application logic of the task at hand, and less on debugging an abandoned method
for sentence segmentation or reimplementing noun phrase chunking.

This presentation will cover a handful of the NLP building blocks provided by
NLTK (and a few additional libraries), including extracting text from HTML,
stemming & lemmatization, frequency analysis, and named entity recognition.
Several of these components will then be assembled to build a very basic
document summarization program.

### Initial Setup
Obviously, you'll need Python installed on your system to run the code examples
used in this presentation. We enthusiatically recommend using
[Anaconda](https://store.continuum.io/cshop/anaconda/), a Python distribution
provided by [Continuum Analytics](http://www.continuum.io/). Anaconda is free to
use, it includes nearly [200 of the most commonly used Python packages for data
analysis](http://docs.continuum.io/anaconda/pkg-docs.html) (including NLTK), and
it works on Mac, Linux, and yes, even Windows.

We'll make use of the following Python packages in the example code:

- [nltk](http://www.nltk.org/install.html) (comes with Anaconda)
- [readability-lxml](https://github.com/buriy/python-readability)
- [BeautifulSoup4](http://www.crummy.com/software/BeautifulSoup/) (comes with
Anaconda)
- [scikit-learn](http://scikit-learn.org/stable/install.html) (comes with
Anaconda)

Please note that the **readability** package is not distributed with Anaconda,
so you'll need to download & install it separately using something like
<code>easy_install readability-lxml</code> or <code>pip install readability-lxml</code>.

If you don't use Anaconda, you'll also need to download & install the other
packages separately using similar methods. Refer to the homepage of each package
for instructions.

You'll want to run <code>nltk.download()</code> one time to get all of the NLTK
packages, corpora, etc. (see below). Select the "all" option. Depending on your
network speed, this could take a while, but you'll only need to do it once.

#### Java libraries (optional)
One of the examples will use NLTK's interface to the [Stanford Named Entity
Recognizer](http://www-nlp.stanford.edu/software/CRF-NER.shtml#Download), which
is distributed as a Java library. In particular, you'll want the following files
handy in order to run this particular example:

- stanford-ner.jar
- english.all.3class.distsim.crf.ser.gz

***

## Getting Started
The first thing we'll need to do is <code>import nltk</code>:


    import nltk

#### Downloading NLTK resources
The first time you run anything using NLTK, you'll want to go ahead and download
the additional resources that aren't distributed directly with the NLTK package.
Upon running the <code>nltk.download()</code> command below, the the NLTK
Downloader window will pop-up. In the Collections tab, select "all" and click on
Download. As mentioned earlier, this may take several minutes depending on your
network connection speed, but you'll only ever need to run it a single time.


    nltk.download()

## Extracting text from HTML
Now the fun begins. We'll start with a pretty basic and commonly-faced task:
extracting text content from an HTML page. Python's urllib package  gives us the
tools we need to fetch a web page from a given URL, but we see that the output
is full of HTML markup that we don't want to deal with.

(N.B.: Throughout the examples in this presentation, we'll use Python *slicing*
(e.g., <code>[:500]</code> below) to only display a small portion of a string or
list. Otherwise, if we displayed the entire item, sometimes it would take up the
entire screen.)


    from urllib import urlopen
    
    url = "http://venturebeat.com/2014/07/04/facebooks-little-social-experiment-got-you-bummed-out-get-over-it/"
    html = urlopen(url).read()
    html[:500]

#### Stripping-out HTML formatting
Fortunately, NTLK provides a method called <code>clean_html()</code> to get the
raw text out of an HTML-formatted string. It's still not perfect, though, since
the output will contain page navigation and all kinds of other junk that we
don't want, especially if our goal is to focus on the body content from a news
article, for example.


    text = nltk.clean_html(html)
    text[:500]

#### Identifying the Main Content
If we just want the body content from the article, we'll need to use two
additional packages. The first is a Python port of a Ruby port of a Javascript
tool called Readability, which pulls the main body content out of an HTML
document and subsequently "cleans it up." The second package, BeautifulSoup, is
a Python library for pulling data out of HTML and XML files. It parses HTML
content into easily-navigable nested data structure. Using Readability and
BeautifulSoup together, we can quickly get exactly the text we're looking for
out of the HTML, (*mostly*) free of page navigation, comments, ads, etc. Now
we're ready to start analyzing this text content.


    from readability.readability import Document
    from bs4 import BeautifulSoup
    
    readable_article = Document(html).summary()
    readable_title = Document(html).title()
    soup = BeautifulSoup(readable_article)
    print '*** TITLE *** \n\"' + readable_title + '\"\n'
    print '*** CONTENT *** \n\"' + soup.text[:500] + '[...]\"'

## Frequency Analysis
Here's a little secret: much of NLP (and data science, for that matter) boils
down to counting things. If you've got a bunch of data that needs *analyzin'*
but you don't know where to start, counting things is usually a good place to
begin. Sure, you'll need to figure out exactly what you want to count, how to
count it, and what to do with the counts, but if you're lost and don't know what
to do, **just start counting**.

Perhaps we'd like to begin (as is often the case in NLP) by examining the words
that appear in our document. To do that, we'll first need to tokenize the text
string into discrete words. Since we're working with English, this isn't so bad,
but if we were working with a non-whitespace-delimited language like Chinese,
Japanese, or Korean, it would be much more difficult.

In the code snippet below, we're using two of NLTK's tokenize methods to first
chop up the article text into sentences, and then each sentence into individual
words. (Technically, we didn't need to use <code>sent_tokenize()</code>, but if
we only used <code>word_tokenize()</code> alone, we'd see a bunch of extraneous
sentence-final punctuation in our output.) By printing each token
alphabetically, along with a count of the number of times it appeared in the
text, we can see the results of the tokenization. Notice that the output
contains some punctuation & numbers, hasn't been loweredcased, and counts
*BuzzFeed* and *BuzzFeed's* separately. We'll tackle some of those issues next.


    tokens = [word for sent in nltk.sent_tokenize(soup.text) for word in nltk.word_tokenize(sent)]
    
    for token in sorted(set(tokens))[:30]:
        print token + ' [' + str(tokens.count(token)) + ']'

#### Word Stemming
[Stemming](http://en.wikipedia.org/wiki/Stemming) is the process of reducing a
word to its base/stem/root form. Most stemmers are pretty basic and just chop
off standard affixes indicating things like tense (e.g., "-ed") and possessive
forms (e.g., "-'s"). Here, we'll use the Snowball stemmer for English, which
comes with NLTK.

Once our tokens are stemmed, we can rest easy knowing that *BuzzFeed* and
*BuzzFeed's* are now being counted together as... *buzzfe*? Don't worry:
although this may look weird, it's pretty standard behavior for stemmers and
won't affect our analysis (much). We also (probably) won't show the stemmed
words to users -- we'll normally just use them for internal analysis or indexing
purposes.


    from nltk.stem.snowball import SnowballStemmer
    
    stemmer = SnowballStemmer("english")
    stemmed_tokens = [stemmer.stem(t) for t in tokens]
    
    for token in sorted(set(stemmed_tokens))[50:75]:
        print token + ' [' + str(stemmed_tokens.count(token)) + ']'

#### Lemmatization

Although the stemmer very helpfully chopped off pesky affixes (and made
everything lowercase to boot), there are some word forms that give stemmers
indigestion, especially *irregular* words. While the process of stemming
typically involves rule-based methods of stripping affixes (making them small &
fast), **lemmatization** involves dictionary-based methods to derive the
canonical forms (i.e., *lemmas*) of words. For example, *run*, *runs*, *ran*,
and *running* all correspond to the lemma *run*. However, lemmatizers are
generally big, slow, and brittle due to the nature of the dictionary-based
methods, so you'll only want to use them when necessary.

The example below compares the output of the Snowball stemmer with the WordNet
lemmatizer (also distributed with NLTK). Notice that the lemmatizer correctly
converts *women* into *woman*, while the stemmer turns *lying* into *lie*.
Additionally, both replace *eyes* with *eye*, but neither of them properly
transforms *told* into *tell*.


    lemmatizer = nltk.WordNetLemmatizer()
    temp_sent = "Several women told me I have lying eyes."
    
    print [stemmer.stem(t) for t in nltk.word_tokenize(temp_sent)]
    print [lemmatizer.lemmatize(t) for t in nltk.word_tokenize(temp_sent)]

#### NLTK Frequency Distributions
Thus far, we've been working with lists of tokens that we're manually sorting,
uniquifying, and counting -- all of which can get to be a bit cumbersome.
Fortunately, NLTK provides a data structure called <code>FreqDist</code> that
makes it more convenient to work with these kinds of frequency distributions.
The code snippet below builds a <code>FreqDist</code> from our list of stemmed
tokens, and then displays the top 25 tokens appearing most frequently in the
text of our article. Wasn't that easy?


    fdist = nltk.FreqDist(stemmed_tokens)
    
    for item in fdist.items()[:25]:
        print item

#### Filtering out Stop Words
Notice in the output above that most of the top 25 tokens are worthless. With
the exception of things like *facebook*, *content*, *user*, and perhaps *emot*
(emotion?), the rest are basically devoid of meaningful information. They don't
really tells us anything about the article since these tokens will appear is
just about any English document. What we need to do is filter out these [*stop
words*](http://en.wikipedia.org/wiki/Stop_words) in order to focus on just the
important material.

While there is no single, definitive list of stop words, NLTK provides a decent
start. Let's load it up and take a look at what we get:


    sorted(nltk.corpus.stopwords.words('english'))[:25]

Now we can use this list to filter-out stop words from our list of stemmed
tokens before we create the frequency distribution. You'll notice in the output
below that we still have some things like punctuation that we'd probably like to
remove, but we're much closer to having a list of the most "important" words in
our article.


    stemmed_tokens_no_stop = [stemmer.stem(t) for t in stemmed_tokens if t not in nltk.corpus.stopwords.words('english')]
    
    fdist2 = nltk.FreqDist(stemmed_tokens_no_stop)
    
    for item in fdist2.items()[:25]:
        print item

## Named Entity Recognition
Another task we might want to do to help identify what's "important" in a text
document is [named entity recogniton (NER)](http://en.wikipedia.org/wiki/Named-
entity_recognition). Also called *entity extraction*, this process involves
automatically extracting the names of persons, places, organizations, and
potentially other entity types out of unstructured text. Building an NER
classifier requires *lots* of annotated training data and some [fancy machine
learning algorithms](http://en.wikipedia.org/wiki/Conditional_random_field), but
fortunately, NLTK comes with a pre-built/pre-trained NER classifier ready to
extract entities right out of the box. This classifier has been trained to
recognize PERSON, ORGANIZATION, and GPE (geo-political entity) entity types.

(At this point, I should include a disclaimer stating [No True Computational
Linguist](http://en.wikipedia.org/wiki/No_true_Scotsman) would ever use a pre-
built NER classifier in the "real world" without first re-training it on
annotated data representing their particular task. So please don't send me any
hate mail -- I've done my part to stop the madness.)

In the example below (inspired by [this gist from Gavin
Hackeling](https://gist.github.com/gavinmh/4735528/) and [this post from John
Price](http://freshlyminted.co.uk/blog/2011/02/28/getting-band-and-artist-names-
nltk/)), we're defining a method to perform the following steps:

- take a string as input
- tokenize it into sentences
- tokenize the sentences into words
- add part-of-speech tags to the words using <code>nltk.pos_tag()</code>
- run this through the NLTK-provided NER classifier using
<code>nltk.ne_chunk()</code>
- parse these intermediate results and return any extracted entities

We then apply this method to a sample sentence and parse the clunky output
format provided by <code>nltk.ne_chunk()</code> (it comes as a
[nltk.tree.Tree](http://www.nltk.org/_modules/nltk/tree.html)) to display the
entities we've extracted. Don't let these nice results fool you -- NER output
isn't always this satisfying. Try some other sample text and see what you get.


    def extract_entities(text):
    	entities = []
    	for sentence in nltk.sent_tokenize(text):
    	    chunks = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sentence)))
    	    entities.extend([chunk for chunk in chunks if hasattr(chunk, 'node')])
    	return entities
    
    for entity in extract_entities('My name is Charlie and I work for Altamira in Tysons Corner.'):
        print '[' + entity.node + '] ' + ' '.join(c[0] for c in entity.leaves())

If you're like me, you've grown accustomed over the years to working with the
[Stanford NER](http://nlp.stanford.edu/software/CRF-NER.shtml) library for Java,
and you're suspicious of NLTK's built-in NER classifier (especially because it
has *chunk* in the name). Thankfully, recent versions of NLTK contain an special
<code>NERTagger</code> interface that enables us to make calls to Stanford NER
from our Python programs, even though Stanford NER is a *Java library* (the
horror!). [Not surprisingly](http://www.yurtopic.com/tech/programming/images
/java-and-python.jpg), the Python <code>NERTagger</code> API is slightly less
verbose than the native Java API for Stanford NER.

To run this example, you'll need to follow the instructions for installing the
optional Java libraries, as outlined in the **Initial Setup** section above.
You'll also want to pay close attention to the comment that says <code># change
the paths below to point to wherever you unzipped the Stanford NER download
file</code>.


    from nltk.tag.stanford import NERTagger
    
    # change the paths below to point to wherever you unzipped the Stanford NER download file
    st = NERTagger('/Users/cgreenba/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
                   '/Users/cgreenba/stanford-ner/stanford-ner.jar', 'utf-8')
    
    for i in st.tag('Up next is Tommy, who works at STPI in Washington.'.split()):
        print '[' + i[1] + '] ' + i[0]

## Automatic Summarization
Now let's try to take some of what we've learned and build something potentially
useful in real life: a program that will [automatically
summarize](http://en.wikipedia.org/wiki/Automatic_summarization) documents. For
this, we'll switch gears slightly, putting aside the web article we've been
working on until now and instead using a corpus of documents distributed with
NLTK.

The Reuters Corpus contains nearly 11,000 news articles about a variety of
topics and subjects. If you've run the <code>nltk.download()</code> command as
previously recommended, you can then easily import and explore the Reuters
Corpus like so:


    from nltk.corpus import reuters
    
    print '** BEGIN ARTICLE: ** \"' + reuters.raw(reuters.fileids()[0])[:500] + ' [...]\"'

Our [painfully simplistic](http://anthology.aclweb.org/P/P11/P11-3014.pdf)
automatic summarization tool will implement the following steps:

- assign a score to each word in a document corresponding to its level of
"importance"
- rank each sentence in the document by summing the individual word scores and
dividing by the number of tokens in the sentence
- extract the top N highest scoring sentences and return them as our "summary"

Sounds easy enough, right? But before we can say "*voila!*," we'll need to
figure out how to calculate an "importance" score for words. As we saw above
with stop words, etc. simply counting the number of times a word appears in a
document will not necessarily tell you which words are most important.

#### Term Frequency - Inverse Document Frequency (TF-IDF)

Consider a document that contains the word *baseball* 8 times. You might think,
"wow, *baseball* isn't a stop word, and it appeared rather frequently here, so
it's probably important." And you might be right. But what if that document is
actually an article posted on a baseball blog? Won't the word *baseball* appear
frequently in nearly every post on that blog? In this particular case, if you
were generating a summary of this document, would the word *baseball* be a good
indicator of importance, or would you maybe look for other words that help
distinguish or differentiate this blog post from the rest?

Context is essential. What really matters here isn't the raw frequency of the
number of times each word appeared in a document, but rather the **relative
frequency** comparing the number of times a word appeared in this document
against the number of times it appeared across the rest of the collection of
documents. "Important" words will be the ones that are generally rare across the
collection, but which appear with an unusually high frequency in a given
document.

We'll calculate this relative frequency using a statistical metric called [term
frequency - inverse document frequency (TF-
IDF)](http://en.wikipedia.org/wiki/Tf%E2%80%93idf). We could implement TF-IDF
ourselves using NLTK, but rather than bore you with the math, we'll take a
shortcut and use the TF-IDF implementation provided by the [scikit-learn](http
://scikit-learn.org/) machine learning library for Python.

#### Building a Term-Document Matrix

We'll use scikit-learn's <code>TfidfVectorizer</code> class to construct a
[term-document matrix](http://en.wikipedia.org/wiki/Document-term_matrix)
containing the TF-IDF score for each word in each document in the Reuters
Corpus. In essence, the rows of this sparse matrix correspond to documents in
the corpus, the columns represent each word in the vocabulary of the corpus, and
each cell contains the TF-IDF value for a given word in a given document.

Inspired by a [computer science lab exercise from Duke University](http://www.cs
.duke.edu/courses/spring14/compsci290/assignments/lab02.html), the code sample
below iterates through the Reuters Corpus to build a dictionary of stemmed
tokens for each article, then uses the <code>TfidfVectorizer</code> and scikit-
learn's own built-in stop words list to generate the term-document matrix
containing TF-IDF scores.


    import datetime, re, sys
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    def tokenize_and_stem(text):
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems
    
    token_dict = {}
    for article in reuters.fileids():
        token_dict[article] = reuters.raw(article)
            
    tfidf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english', decode_error='ignore')
    print 'building term-document matrix... [process started: ' + str(datetime.datetime.now()) + ']'
    sys.stdout.flush()
    
    tdm = tfidf.fit_transform(token_dict.values()) # this can take some time (about 60 seconds on my machine)
    print 'done! [process finished: ' + str(datetime.datetime.now()) + ']'

#### TF-IDF Scores

Now that we've built the term-document matrix, we can explore its contents:


    from random import randint
    
    feature_names = tfidf.get_feature_names()
    print 'TDM contains ' + str(len(feature_names)) + ' terms and ' + str(tdm.shape[0]) + ' documents'
    
    print 'first term: ' + feature_names[0]
    print 'last term: ' + feature_names[len(feature_names) - 1]
    
    for i in range(0, 4):
        print 'random term: ' + feature_names[randint(1,len(feature_names) - 2)]

#### Generating the Summary

That's all we'll need to produce a summary for any document in the corpus. In
the example code below, we start by randomly selecting an article from the
Reuters Corpus. We iterate through the article, calculating a score for each
sentence by summing the TF-IDF values for each word appearing in the sentence.
We normalize the sentence scores by dividing by the number of tokens in the
sentence (to avoid bias in favor of longer sentences). Then we sort the
sentences by their scores, and return the highest-scoring sentences as our
summary. The number of sentences returned corresponds to roughly 20% of the
overall length of the article.

Since some of the articles in the Reuters Corpus are rather small (i.e., a
single sentence in length) or contain just raw financial data, some of the
summaries won't make sense. If you run this code a few times, however, you'll
eventually see a randomly-selected article that provides a decent demonstration
of this simplistic method of identifying the "most important" sentence from a
document.


    import math
    from __future__ import division
    
    article_id = randint(0, tdm.shape[0] - 1)
    article_text = reuters.raw(reuters.fileids()[article_id])
    
    sent_scores = []
    for sentence in nltk.sent_tokenize(article_text):
        score = 0
        sent_tokens = tokenize_and_stem(sentence)
        for token in (t for t in sent_tokens if t in feature_names):
            score += tdm[article_id, feature_names.index(token)]
        sent_scores.append((score / len(sent_tokens), sentence))
    
    summary_length = int(math.ceil(len(sent_scores) / 5))
    sent_scores.sort(key=lambda sent: sent[0])
    
    print '*** SUMMARY ***'
    for summary_sentence in sent_scores[:summary_length]:
        print summary_sentence[1]
    
    print '\n*** ORIGINAL ***'
    print article_text

#### Improving the Summary
That was fairly easy, but how could we improve the quality of the generated
summary? Perhaps we could boost the importance of words found in the title or
any entities we're able to extract from the text. After initially selecting the
highest-scoring sentence, we might discount the TF-IDF scores for duplicate
words in the remaining sentences in an attempt to reduce repetitiveness. We
could also look at cleaning up the sentences used to form the summary by fixing
any pronouns missing an antecedent, or even pulling out partial phrases instead
of complete sentences. The possibilities are virtually endless.

## Next Steps
Want to learn more? Start by working your way through all the examples in the
NLTK book (aka "the Whale book"):

- [Natural Language Processing with Python
(book)](http://oreilly.com/catalog/9780596516499/)
- (free online version: [nltk.org/book](http://www.nltk.org/book/))

### Additional NLP Resources for Python
- [NLTK HOWTOs](http://www.nltk.org/howto/)
- [Python Text Processing with NLTK 2.0 Cookbook (book)](http://www.packtpub.com
/python-text-processing-nltk-20-cookbook/book)
- [Python wrapper for the Stanford CoreNLP Java
library](https://pypi.python.org/pypi/corenlp)
- [guess_language (Python library for language
identification)](https://bitbucket.org/spirit/guess_language)
- [MITIE (new C/C++-based NER library from MIT with a Python
API)](https://github.com/mit-nlp/MITIE)
- [gensim (topic modeling library for Python)](http://radimrehurek.com/gensim/)

### Attend future DC NLP meetups

- [dcnlp.org](http://dcnlp.org/) | [@DCNLP](https://twitter.com/DCNLP/)
