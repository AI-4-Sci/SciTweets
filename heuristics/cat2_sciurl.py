import pandas as pd
from tqdm import tqdm
import requests
import tldextract
tqdm.pandas()

#import os
#os.getcwd()
#os.chdir('labeling/scidiscourse_heuristics')

subdomains = pd.read_csv('repo_subdomains.csv')['domain'].values
sci_mags_domains = pd.read_csv('science_mags_domains.csv')['domain'].values
sci_news_domains = pd.read_csv('news_outlets_domains.csv')['domain'].values


def url2domain(url, retry=False, resolve=False):
    """Extract the domain from URL.
    """

    url = str(url)

    if resolve:
        if "bit.ly" in url or "doi.org" in url:
            try:  # try to resolve bit.ly and doi.org links, if fails continue
                s = requests.Session()
                s.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 Edg/98.0.1108.55"}
                r = s.head(url, allow_redirects=True, timeout=2)
                url = r.url
            except Exception:
                pass

    res = tldextract.extract(url)

    domain_parts = []
    subdomain_parts = []

    if res.domain != '':
        domain_parts.append(res.domain)

    if res.suffix != '':
        domain_parts.append(res.suffix)

    if res.subdomain != '':
        subdomain_parts.append(res.subdomain)

    subdomain_parts.extend(domain_parts)

    if len(domain_parts) >= 2:
        return url, res.suffix, ".".join(domain_parts), ".".join(subdomain_parts)
    else:
        if retry:
            try:  # try to resolve faulty link, if fails continue (e.g. http://gov.uk has only gov.uk suffix but no domain -> faulty, resolving to www.gov.uk -> domain = www)
                s = requests.Session()
                s.headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 Edg/98.0.1108.55"}
                r = s.head(url, allow_redirects=True, timeout=2)
                url = r.url
                return url2domain(url, retry=False, resolve=resolve)
            except:
                pass

        return url, "error", "error", "error"


def prepare_urls(tweets):
    # split tweets into tweets with url and tweets without url
    tweets_w_url = tweets[~tweets['urls'].isna()].copy()
    tweets_wo_url = tweets[tweets['urls'].isna()].copy()

    # annotate if a tweet contains a url
    tweets_w_url['has_url'] = True
    tweets_wo_url['has_url'] = False

    # for tweets with url, explode tweets with multiple urls to multiple rows
    tweets_w_url["urls"] = tweets_w_url["urls"].str.split(":-:")
    tweets_w_url = tweets_w_url.explode("urls").reset_index(drop=True)
    tweets_w_url = tweets_w_url[tweets_w_url["urls"] != ""]
    tweets_w_url = tweets_w_url.rename(columns={'urls': 'url'})

    # for tweets with url, annotate each url with tld, domain and subdomain
    res = tweets_w_url['url'].apply(url2domain)
    res = list(map(list, zip(*res.values)))
    tweets_w_url["processed_url"] = res[0]
    tweets_w_url["tld"] = res[1]
    tweets_w_url["tld"] = tweets_w_url["tld"].str.lower()
    tweets_w_url["domain_tld"] = res[2]
    tweets_w_url["domain_tld"] = tweets_w_url["domain_tld"].str.lower()
    tweets_w_url["subdomain_domain_tld"] = res[3]
    tweets_w_url["subdomain_domain_tld"] = tweets_w_url["subdomain_domain_tld"].str.lower()

    # for tweets with url, group tweets with multiple urls to back to single row
    cols = tweets_w_url.columns.tolist()
    cols = [col for col in cols if col not in ['url', 'processed_url', 'tld', 'domain_tld', 'subdomain_domain_tld']]
    tweets_w_url = (tweets_w_url.groupby(by=cols, dropna=False)
                    .agg({'url': lambda x: x.tolist(),
                          'processed_url': lambda x: x.tolist(),
                          'tld': lambda x: x.tolist(),
                          'domain_tld': lambda x: x.tolist(),
                          'subdomain_domain_tld': lambda x: x.tolist()
                          })
                    .rename(columns={'url': 'urls',
                                     'processed_url': 'processed_urls',
                                     'tld': 'tlds',
                                     'domain_tld': 'domain_tlds',
                                     'subdomain_domain_tld': 'subdomain_domain_tlds'
                                     })
                    .reset_index())

    # fill empty fields for tweets without urls
    tweets_wo_url["urls"] = [[]] * len(tweets_wo_url)
    tweets_wo_url["processed_urls"] = [[]] * len(tweets_wo_url)
    tweets_wo_url["tlds"] = [[]] * len(tweets_wo_url)
    tweets_wo_url["domain_tlds"] = [[]] * len(tweets_wo_url)
    tweets_wo_url["subdomain_domain_tlds"] = [[]] * len(tweets_wo_url)

    tweets = pd.concat([tweets_w_url ,tweets_wo_url])
    return tweets


def annotate_sci_crossref_subdomains(domains):
    matches = []

    for domain in domains:
        for sci_domain in subdomains:
            if sci_domain in domain:
                if domain == sci_domain:
                    matches.append(domain)
                # check if subdomain is below sci_subdomain
                elif domain.endswith('.' + sci_domain):
                    matches.append(domain)

    return "; ".join(matches)


def annotate_sci_mag_domains(domains):
    matches = []

    for domain in domains:
        for sci_domain in sci_mags_domains:
            if sci_domain in domain:
                if domain == sci_domain:
                    matches.append(domain)
                # check if subdomain is below sci_subdomain
                elif domain.endswith('.' + sci_domain):
                    matches.append(domain)

    return "; ".join(matches)


def annotate_sci_news_domains(domains, urls):
    matches = []

    for domain, url in zip(domains, urls):
        if "/science" in url:
            for sci_domain in sci_news_domains:
                if sci_domain in domain:
                    if domain == sci_domain:
                        matches.append(domain)
                    # check if subdomain is below sci_subdomain
                    elif domain.endswith('.' + sci_domain):
                        matches.append(domain)

    return "; ".join(matches)


def annotate_tweets(tweets):
    tweets_w_url = tweets[tweets['has_url']].copy()
    tweets_wo_url = tweets[~tweets['has_url']].copy()

    # check if tweet subdomains are science-related
    tweets_w_url['sci_subdomain'] = tweets_w_url['subdomain_domain_tlds'].progress_apply(annotate_sci_crossref_subdomains)
    tweets_w_url["has_sci_subdomain"] = tweets_w_url['sci_subdomain'].apply(lambda x: True if len(x) > 0 else False)

    tweets_wo_url['sci_subdomain'] = tweets_wo_url['has_url'].apply(lambda x: [])
    tweets_wo_url["has_sci_subdomain"] = False

    tweets_w_url['sci_mag_domain'] = tweets_w_url['subdomain_domain_tlds'].progress_apply(annotate_sci_mag_domains)
    tweets_w_url["has_sci_mag_domain"] = tweets_w_url['sci_mag_domain'].apply(lambda x: True if len(x) > 0 else False)

    tweets_wo_url['sci_mag_domain'] = tweets_wo_url['has_url'].apply(lambda x: [])
    tweets_wo_url["has_sci_mag_domain"] = False

    tweets_w_url['sci_news_domain'] = tweets_w_url[['subdomain_domain_tlds', 'processed_urls']].progress_apply(lambda x: annotate_sci_news_domains(x[0], x[1]), axis=1)
    tweets_w_url["has_sci_news_domain"] = tweets_w_url['sci_news_domain'].apply(lambda x: True if len(x) > 0 else False)

    tweets_wo_url['sci_news_domain'] = tweets_wo_url['has_url'].apply(lambda x: [])
    tweets_wo_url["has_sci_news_domain"] = False

    tweets = pd.concat([tweets_w_url, tweets_wo_url], ignore_index=True)
    return tweets


if __name__ == '__main__':

    # tweets dataframe need a "urls" column that is either nan or contains the tweets urls concatenated using ":-:"
    tweets = pd.read_csv('...', sep='\t')

    print('prepare tweets with urls (extract subdomains)')
    tweets = prepare_urls(tweets)

    print('run heuristics')
    tweets = annotate_tweets(tweets)
     
    tweets['is_cat2'] = tweets[['has_sci_subdomain', 'has_sci_mag_domain', 'has_sci_news_domain']].any(axis='columns')
    
    cat2_tweets = tweets[tweets['is_cat2']]