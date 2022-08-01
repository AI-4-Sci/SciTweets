# Heuristics
This directory contains code and supplementary material for the heuristics described in section 2.2 of the paper.

__Table of contents:__
- [Heuristics for Category 1.1](#heuristics-for-category-11)
- [Heuristics for Category 1.2](#heuristics-for-category-12)
- [Heuristics for Category 1.3](#heuristics-for-category-13)

## Heuristics for Category 1.1 (Scientific Knowledge (scientifically verifiable claims/questions))
======================================== <br/>
To find tweets that are scientifically verifiable claims or questions, we develop two different heuristics:
1. Heuristic 1: pattern-matching for subject-predicate-object patterns (more specifically: noun-verb-noun and noun-verb-adjective patterns) where the predicate must come from a list of predefined predicates (e.g « cause », « lead to », « help with ») that we extracted from different research works on claims [1-4].
2. Heuristic 2: scientific term filter where we only keep tweets that contain at least one term from a predefined list of ~30k scientific terms that come from Wikipedia Glossaries from which we hand-picked the categories that we deemed scientific (e.g « medicine », « history », « biology »). 

We then combine the two heuristics to only select tweets that match our defined pattern, contain a scientific predicate as a verb, and contain a scientific term. 

## Heuristics for Category 1.2 (Reference to Scientific Knowledge)
======================================== <br/>
To find tweets that most probably contain references to Scientific Knowledge we filter tweets that contain a URL with a subdomain that is included in a
predefined list of 17,500 scientific domains and subdomains from open access
repositories, newspaper science sections and science magazines
(e.g., “link.springer.com“, “sciencedaily.com“).

Number of scientific domains and subdomains per type:

| Type                       | Number of domains and subdomains |
|----------------------------|:--------------------------------:|
| Open Access Repositories   |              17,463              |
| Newspaper Science Sections |                23                |
| Science Magazines          |                14                |
| Total                      |              17,500              |

####Open Access Repositories 
The list of 17,463 subdomains is collected as follows:
1. all full-text links included in the public CrossRef Snapshot from January 2021 ([Link](https://academictorrents.com/details/e4287cb7619999709f6e9db5c359dda17e93d515)) were extracted
2. using the CrossRef API we extracted the full-text links that were registered after January 2021
3. all extracted links were annotated with their subdomain using the python library [tldextract](https://github.com/john-kurkowski/tldextract)
4. the frequency for every subdomain was computed  
5. we excluded subdomains with a frequency lower than 50
6. we excluded 38 subdomains that are clearly not scientific (e.g., "youtube.com", "yahoo.com")
7. we added 46 subdomains from a manually curated list (e.g., "semanticscholar.org", "www.biorxiv.org")

To filter tweets that refer to Open Access Repositories, the tweets must contain a URL with a subdomain from this list. 

#### Newspaper Science Sections
The list of 23 Newspaper Science Sections is manually curated and contains domains from major newspaper outlets in english language, that have a dedicated science section.
To filter tweets that refer to Newspaper Science Sections, the tweets must contain a URL with a subdomain from this list **AND** includes "/science".

####Science Magazines
The list of 14 Science Magazines domains and subdomains is manually curated.
To filter tweets that refer to Science Magazines, the tweets must contain a URL with a subdomain from this list.

## Heuristics for Category 1.3 (Related to scientific research in general)
======================================== <br/>
To find tweets that are related to scientific research, we develop 5 different heuristics:
1. Heuristic 1: includes tweets that mention scientists, i.e., that have a noun from a predefined list, e.g., "research team", "research group", "scientist", "researcher", "psychologist", "biologist", "economist"
2. Heuristic 2: includes tweets that mention research, i.e., that have a noun from a predefined list, e.g., 'research on', 'research in', 'research for', 'research from', 'science of', 'science to', 'science', 'sciences of'
3. Heuristic 3: tweets that mention a research method, i.e., that have a word from a predefined list of social science methods, collected from SAGE Social Science Thesaurus (see sc_methods.txt) [Link](https://concepts.sagepub.com/vocabularies/social-science/en/page/?uri=https%3A%2F%2Fconcepts.sagepub.com%2Fsocial-science%2Fconcept%2Fconceptgroup%2Fmethods)
4. Heuristic 4: includes tweets that mention research outcomes, i.e., that have a noun from a predefined list, e.g.,'publications', 'posters', 'reports', 'statistics', 'datasets', 'findings'

All 4 heuristics were combined with a logical OR. 

## References
======================================== <br/>
- [1] Pinto, J. M. G., Wawrzinek, J., & Balke, W. T. (2019, June). What Drives Research Efforts? Find Scientific Claims that Count!. In 2019 ACM/IEEE Joint Conference on Digital Libraries (JCDL) (pp. 217-226). IEEE.
- [2] González Pinto, J. M., & Balke, W. T. (2018, September). Scientific claims characterization for claim-based analysis in digital libraries. In International Conference on Theory and Practice of Digital Libraries (pp. 257-269). Springer, Cham.
- [3] Kilicoglu, H., Rosemblat, G., Fiszman, M., & Rindflesch, T. C. (2011). Constructing a semantic predication gold standard from the biomedical literature. BMC bioinformatics, 12(1), 1-17.
- [4] Smeros, P., Castillo, C., & Aberer, K. (2021, October). SciClops: Detecting and Contextualizing Scientific Claims for Assisting Manual Fact-Checking. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 1692-1702).
