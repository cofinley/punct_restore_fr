from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

tokenizer = AutoTokenizer.from_pretrained("cfinley/punct_restore_fr")

model = AutoModelForTokenClassification.from_pretrained("cfinley/punct_restore_fr")
nlp = pipeline('ner', model=model, tokenizer=tokenizer)

sentence = '''
cette vidéo a été tournée juste avant le
confinement est maintenant reste chez
nous et vous restez chez vous aussi
comme ça vous rester en vie bonne vidéo
vous allez voir les excellentes
minecraft le jeu le plus primé sur
youtube 170 millions de ventes
une licence grandiose complexe apprécié
par des gamer et gamers du monde entier
et devinez qui sont les deux couillons
qui vont le découvrir alors qu'ils sont
'''.replace('\n', ' ').strip()

predictions = nlp(sentence)

boundaries = [p['start']-1 for p in predictions[1:]]
boundaries.insert(0, 0)

sents = []
for i in range(1, len(boundaries)):
    sent = sentence[boundaries[i-1]:boundaries[i]].strip()
    sents.append(sent)
sent = sentence[boundaries[-1]:].strip()
sents.append(sent)

print('\n\n'.join(sents))