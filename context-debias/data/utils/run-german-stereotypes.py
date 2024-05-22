from german_nouns.lookup import Nouns
from pprint import pprint
import random

BASE_PATH = "/home/viktorija/bakalaurinis/context-debias/data/german/stereotypes.txt"

nouns = Nouns()    
m_stereotypes = set()
f_stereotypes = set()

with open(BASE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip().capitalize()
        print(word)
        if nouns[word]:
            entry = nouns[word]
            noun = None

            for e in entry:
                if 'genus' in e.keys() and (e['genus'] == 'm' or e['genus'] == 'f'):
                    if e['genus'] == 'm':
                        m_stereotypes.add(word)
                    elif e['genus'] == 'f':
                        f_stereotypes.add(word)
                    noun = e
                    break

# Make sets same length
if len(m_stereotypes) > len(f_stereotypes):
    m_stereotypes = list(m_stereotypes)[:len(f_stereotypes)]
    new_stereotypes = m_stereotypes + list(f_stereotypes)
else:
    f_stereotypes = list(f_stereotypes)[:len(m_stereotypes)]
    new_stereotypes = f_stereotypes + list(m_stereotypes)

random.shuffle(new_stereotypes)

print('m len:', len(m_stereotypes))
print('f len:', len(f_stereotypes))
print('new len:', len(new_stereotypes))

# Save the new stereotypes
with open(BASE_PATH.replace('.txt', '_new.txt'), 'w', encoding='utf-8') as f:
    for word in new_stereotypes:
        f.write(word.capitalize() + '\n')
