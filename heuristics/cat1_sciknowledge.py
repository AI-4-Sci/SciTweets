import re
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import en_core_web_sm
from tqdm import tqdm
tqdm.pandas()
nlp = en_core_web_sm.load()

# Predicates word list
predicates = ['affect', 'affects',
              'are a', 'associated with',
              'cause', 'causes', 'correlated with',
              'decrease', 'decreases', 'diminish', 'diminishes',
              'enable', 'enables',
              'facilitate', 'facilitates',
              'higher than', 'hinder', 'hinders',
              'increase', 'increases', 'increment','increments','inhibit', 'inhibits',
              'lead to', 'leads to', 'lower', 'lower than', 'lowers',
              'need less', 'need more', 'needs less', 'needs more',
              'prevent', 'prevents', 'process of', 'promote', 'promotes',
              'reason for', 'reason why',
              'stop', 'stops', 'support', 'supports', 'treat', 'treats']

def load_scientific_terms():
    with open('wiki_sci_terms.txt', 'r') as f:
        wiki_sci_terms = [line.strip() for line in f]

    with open('sc_methods.txt', 'r') as f:
        sc_methods_terms = [line.strip() for line in f]

    scientific_terms = wiki_sci_terms + sc_methods_terms
    scientific_terms.sort()

    for i in range(len(scientific_terms)):
        scientific_terms[i] = scientific_terms[i].lower()

    # Text speak words like "lol" are collected then filtered out of the scientific terms list
    text_speak_wordlist = ['lol','lmao','lmfao', 'omg', 'ppl', 'pls','thx','sry','dah', 'dat', 'soo', 'isn', 'stan','abt', 'dawg','cha','jk']
    # 500 false scientific words are collected then taken away from the scientific terms list, by a process of using the sciterm heuristic and looking at the
    # most frequent words collected
    false_scientific_words_list = ['haven', 'practice','fans','call','place','special','account','young','small','point','games','act','join','past','words','events','respect','simple','current','student','major','account','cover','moment', 'direct','complete','five','cute','club','non','final','voice','essential','date','tour','ads','till','card','youtube','write','pitch','interviews','duo','cant','even','community', 'force', 'support', 'even', 'food', 'associated', 'death', 'say', 'author', 'find', 'reading', 'being', 'work', 'air', 'storms', 'security', 'real', 'say', 'kissing', 'day', 'need', 'support', 'feedback', 'being', 'perfect', 'religious', 'curriculum', 'good', 'line', 'schools', 'women', 'link', 'meet', 'mind', 'eyes', 'crew', 'support', 'being', 'work', 'ago', 'load', 'associated', 'clean', 'first', 'insurance', 'year', 'end', 'good', 'lie', 'support', 'nah', 'rain', 'good', 'share', 'time', 'support', 'support', 'bird', 'bus', 'grade', 'groups', 'good', 'linked', 'negative', 'self', 'number', 'real', 'support', 'satellites', 'tap', 'wire', 'world', 'support', 'problem', 'range', 'resource', 'support', 'day', 'support', 'trade', 'almost', 'authority', 'symptoms', 'women', 'strength', 'support', 'centre', 'footprint', 'sets', 'groups', 'higher', 'resolution', 'series', 'volatile', 'fly', 'support', 'minor', 'first', 'time', 'tax', 'game', 'associated', 'good', 'year', 'markets', 'symptoms', 'driver', 'lesson', 'support', 'women', 'support', 'support', 'world', 'sink', 'time', 'change', 'cost', 'green', 'scope', 'speed', 'key', 'angles', 'end', 'scheme', 'free', 'support', 'message', 'carrier', 'communications', 'support', 'package', 'cactus', 'fruit', 'being', 'full', 'support', 'strong', 'action', 'good', 'free', 'hotspot', 'internet', 'support', 'local', 'support', 'entertainment', 'bit', 'groups', 'support', 'need', 'value', 'emotions', 'negative', 'night', 'support', 'complex', 'freezing', 'province', 'rain', 'snow', 'killing', 'plan', 'global', 'kind', 'movement', 'message', 'support', 'schools', 'peace', 'prayer', 'pressure', 'cars', 'support', 'demand', 'food', 'free', 'effectiveness', 'food', 'medications', 'design', 'ear', 'vibration', 'debt', 'service', 'head', 'school', 'line', 'local', 'internal', 'support', 'reason', 'web', 'negative', 'find', 'higher', 'zoning', 'collection', 'compliance', 'catastrophic', 'control', 'stand', 'support', 'learning', 'mental', 'students', 'support', 'entity', 'laws', 'even', 'process', 'year', 'family', 'spread', 'support', 'understanding', 'want', 'positive', 'good', 'note', 'time', 'top', 'being', 'activity', 'higher', 'active', 'burn', 'change', 'color', 'red', 'aging', 'being', 'fix', 'processes', 'eventually', 'label', 'energy', 'need', 'support', 'time', 'change', 'journey', 'related', 'support', 'fire', 'stream', 'being', 'expression', 'free', 'speaking', 'group', 'member', 'support', 'class', 'classes', 'community', 'game', 'grade', 'play', 'school', 'second', 'skills', 'students', 'time', 'matter', 'world', 'child', 'greedy', 'sound', 'support', 'interaction', 'leadership', 'member', 'position', 'blocking', 'need', 'reason', 'support', 'case', 'list', 'community', 'depreciation', 'assembly', 'support', 'total', 'behaviors', 'change', 'negative', 'positive', 'higher', 'world', 'year', 'industry', 'support', 'activists', 'century', 'food', 'critical', 'mental', 'positive', 'thinking', 'thought', 'energy', 'support', 'country', 'site', 'group', 'front', 'support', 'energy', 'higher', 'independence', 'prices', 'industry', 'year', 'sign', 'face', 'problem', 'support', 'world', 'want', 'information', 'matrix', 'power', 'prayer', 'reason', 'say', 'world', 'back', 'city', 'energy', 'habit', 'support', 'hair', 'event', 'first', 'link', 'second', 'support', 'child', 'governor', 'support', 'age', 'children', 'game', 'groups', 'reverse', 'training', 'touch', 'back', 'culture', 'good', 'need', 'real', 'time', 'annual', 'spring', 'real', 'women', 'need', 'relief', 'support', 'gesture', 'support', 'enough', 'harvesting', 'interference', 'enough', 'household', 'need', 'range', 'discussion', 'information', 'patent', 'support', 'data', 'group', 'groups', 'lots', 'span', 'year', 'being', 'list', 'top', 'landslide', 'support', 'bit', 'killing', 'channel', 'end', 'ran', 'standards', 'revolution', 'absolutely', 'entire', 'need', 'thread', 'digital', 'support', 'back', 'situation', 'breaking', 'industry', 'path', 'support', 'music', 'participation', 'similar', 'face', 'support', 'role', 'product', 'big', 'primary', 'significant', 'support', 'mad', 'being', 'path', 'link', 'group', 'handle', 'need', 'school', 'students', 'support', 'teachers', 'country', 'day', 'source', 'vital', 'class', 'want', 'wealth', 'work', 'end', 'host', 'focus', 'root', 'support', 'time', 'world', 'students', 'support', 'add', 'bar', 'bit', 'couple', 'film', 'set', 'star', 'support', 'motor', 'city', 'support', 'cut', 'finds', 'play']
    scientific_terms = [term for term in scientific_terms if (term not in text_speak_wordlist) and (term not in false_scientific_words_list) ]
    return scientific_terms

# SCI-HEURISTICS

# HEURISTIC 1: CONTAINS ARGUMENTATIVE RELATION
def contains_arg_relation(tweet_sentence):
    for pred in predicates:
        if re.match('.*\s('+pred+')\s.{2,}', tweet_sentence) is not None:
            return pred
    return ""


# HEURISTIC 2: CONTAINS SCIENTIFIC TERM
def contains_scientific_term(tweet_sentence, one_gram_sciterm_only=True):
    # Option 1: Check only for 1-gram sciterms in tweets
    if one_gram_sciterm_only:
        sciterms = []
        tweet_tokens = word_tokenize(tweet_sentence)
        for sciterm in scientific_terms:
            if sciterm in tweet_tokens:
                sciterms.append(sciterm)
        if len(sciterms) > 0:
            return True, sciterms
        return False, []
    # Option 2: Check for all n-gram sciterms in tweets
    # WARNING: IMPERFECT, DO NOT USE OPTION 2 FOR NOW
    # EXPLANATION = (sciterm="top"):no difference between a good positive match ("top .."->matches "top"+" ")
    # and a bad positive match ("stop .."-> also matches "top"+" "
    else:
        sciterms = []
        for sciterm in scientific_terms:
            if (' '+sciterm+' ' in tweet_sentence) or (sciterm+' ' in tweet_sentence) or (' '+sciterm in tweet_sentence):
                sciterms.append(sciterm)
        if len(sciterms) > 0:
            return True, sciterms
        return False, []

# HEURISTIC 3: IS CLAIM (PATTERN-MATCHING: NOUN + PRED + (NOUN OR ADJ) where PRED = a predicate from the list of predicates)
def is_claim(tweet):
    tweet = tweet.lower()
    pred = contains_arg_relation(tweet)
    if pred != "":
        sentences = sent_tokenize(tweet)

        for sent in sentences:
            doc = nlp(sent)

            if " "+pred+" " in sent:
                tags = [token.tag_ for token in doc]
                poss = [token.pos_ for token in doc]
                ents = [token.ent_type_ for token in doc]
                texts = [token.lower_ for token in doc]

                if len(pred.split(" ")) > 1:
                    pred_index = texts.index(pred.split(" ")[0])
                else:
                    pred_index = texts.index(pred)

                #if (pred == "support" and poss[pred_index] != 'NOUN') or pred != "support":
                tags_before = tags[:pred_index]
                poss_before = poss[:pred_index]
                ents_before = ents[:pred_index]

                tags_after = tags[pred_index+1:]
                poss_after = poss[pred_index+1:]
                ents_after = ents[pred_index+1:]


                # Looked for pattern = NOUN + PRED + (NOUN OR ADJ)
                # Condition = what's before the predicate IS a noun AND IS NOT one of the following: personal pronoun, possessive pronoun, person including fictional
                if 'PRP' not in tags_before and 'PRP$' not in tags_before and 'PERSON' not in ents_before and 'NOUN' in poss_before:
                    # Same condition for what's after the predicate
                    if 'PRP' not in tags_after and 'PRP$' not in tags_after and 'PERSON' not in ents_after and ('NOUN' or 'ADJ' in poss_after):
                        if "?" in sent:
                            if " how " in sent or "when " in sent or "why " in sent:
                                return True, 'claim_question', sent
                            else:
                                return True, 'question', sent
                        else:
                            return True, pred, sent

    return False, "", ""

# COMPOUND HEURISTICS
def is_claim_with_sciterm(tweet):
    return is_claim(tweet)[0] and contains_scientific_term(tweet, one_gram_sciterm_only=True)[0]


def annotate_tweets(tweets):
    res = tweets['text'].progress_apply(is_claim_with_sciterm)
    tweets['is_cat1'] = res

    return tweets

if __name__ == '__main__':

    tweets = pd.read_csv('...', sep='\t')

    print('run heuristics')
    tweets = annotate_tweets(tweets)

    cat1_tweets = tweets[tweets['is_cat1']]
