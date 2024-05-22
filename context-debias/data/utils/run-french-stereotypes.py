from pylexique import Lexique383
from pprint import pprint
import random

BASE_PATH = "/home/viktorija/bakalaurinis/context-debias/data/french/stereotypes.txt"

LEXIQUE = Lexique383()
m_stereotypes = set()
f_stereotypes = set()

with open(BASE_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        word = line.strip()
        if word in LEXIQUE.lexique:
            entry = LEXIQUE.get_lex(word)
            noun = None

            if not isinstance(entry, list):
                entry = [entry]
            for e in entry:
                if e.cgram == 'NOM' and (e.genre == 'm' or e.genre == 'f'):
                    if e.genre == 'm':
                        m_stereotypes.add(e.ortho)
                    elif e.genre == 'f':
                        f_stereotypes.add(e.ortho)
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

# Save the new stereotypes
with open(BASE_PATH.replace('.txt', '_new.txt'), 'w', encoding='utf-8') as f:
    for word in new_stereotypes:
        f.write(word + '\n')

