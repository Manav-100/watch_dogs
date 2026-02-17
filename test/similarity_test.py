import nltk
from nltk.corpus import wordnet as wn
import pandas as pd

# --- Step 1: Download WordNet Database ---
print("Downloading WordNet data (this may take a moment)...")
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
print("Download complete. Starting extraction...")

# --- Step 2: Define Helper Functions ---

def get_pos_set(word):
    """
    Checks if a word exists in WordNet as a Noun, Verb, Adj, or Adv.
    """
    pos_set = set()
    if wn.synsets(word, pos=wn.NOUN): pos_set.add('Noun')
    if wn.synsets(word, pos=wn.VERB): pos_set.add('Verb')
    if wn.synsets(word, pos=wn.ADJ) or wn.synsets(word, pos=wn.ADJ_SAT): pos_set.add('Adj')
    if wn.synsets(word, pos=wn.ADV): pos_set.add('Adv')
    
    # Heuristics for words like "running" (Participles)
    if word.endswith('ing'): pos_set.add('Verb')
    
    return pos_set

def generate_formatted_meaning(w1, w2, pattern):
    """
    Creates the structured 'meaning' phrase (e.g., 'a bird that is blue').
    """
    w1_clean = w1.replace('_', ' ')
    w2_clean = w2.replace('_', ' ')
    
    # Determine proper article (a/an)
    article = "an" if w2_clean[0].lower() in 'aeiou' else "a"
    article_w1 = "an" if w1_clean[0].lower() in 'aeiou' else "a"

    definition = ""

    # --- NOUN PATTERNS ---
    if pattern == 'Adj + Noun':
        # "blue_bird" -> "a bird that is blue"
        definition = f"{article} {w2_clean} that is {w1_clean}"

    elif pattern == 'Verb + Noun':
        # "running_shoes" -> "shoes for running"
        if w1.endswith('ing'):
            definition = f"{article} {w2_clean} for {w1_clean}"
        else:
            # "drift_wood" -> "wood that drifts"
            verb = w1_clean
            if not verb.endswith('s'): verb += "s"
            definition = f"{article} {w2_clean} that {verb}"

    elif pattern == 'Noun + Verb':
        # "sun_rise" -> "a sun that rises"
        verb = w2_clean
        if not verb.endswith('s'): verb += "s"
        definition = f"{article} {w1_clean} that {verb}"
        
    elif pattern == 'Noun + Adj':
        # "knight_errant" -> "a knight that is errant"
        definition = f"{article} {w1_clean} that is {w2_clean}"

    # --- VERB PATTERNS ---
    elif pattern == 'Phrasal Verb (Verb + Adv)':
        # "give_up" -> "to give up"
        definition = f"to {w1_clean} {w2_clean}"

    elif pattern == 'Phrasal Verb (Verb + Noun)':
        # "make_love" -> "to make love"
        definition = f"to {w1_clean} {w2_clean}"
        
    elif pattern == 'Phrasal Verb (Noun + Verb)':
        # "baby_sit" -> "to sit a baby"
        definition = f"to {w2_clean} {article_w1} {w1_clean}"

    return definition

# --- Step 3: Main Extraction Logic ---

def extract_all_combinations():
    results = []
    seen_phrases = set()
    
    # We loop through both Noun and Verb synsets
    target_pos = {'n': 'Noun', 'v': 'Verb'}

    for pos_key, pos_type in target_pos.items():
        print(f"Scanning {pos_type}s...")
        
        for synset in wn.all_synsets(pos_key):
            for lemma in synset.lemmas():
                phrase = lemma.name()
                
                # Filter for exactly 2-word phrases (WordNet uses underscores)
                if phrase.count('_') == 1:
                    parts = phrase.split('_')
                    w1, w2 = parts[0].lower(), parts[1].lower()
                    
                    # Skip numbers or symbols
                    if not w1.isalpha() or not w2.isalpha(): continue
                    
                    # Avoid duplicates
                    if phrase in seen_phrases: continue

                    # Check Part of Speech of individual words
                    pos1 = get_pos_set(w1)
                    pos2 = get_pos_set(w2)
                    
                    pattern = None

                    # --- LOGIC FOR NOUN COMPOUNDS ---
                    if pos_type == 'Noun':
                        if 'Adj' in pos1 and 'Noun' in pos2:
                            pattern = 'Adj + Noun'
                        elif 'Verb' in pos1 and 'Noun' in pos2:
                            pattern = 'Verb + Noun'
                        elif 'Noun' in pos1 and 'Verb' in pos2:
                            pattern = 'Noun + Verb'
                        elif 'Noun' in pos1 and 'Adj' in pos2:
                            pattern = 'Noun + Adj'

                    # --- LOGIC FOR VERB COMPOUNDS ---
                    elif pos_type == 'Verb':
                        if 'Verb' in pos1 and 'Adv' in pos2:
                            pattern = 'Phrasal Verb (Verb + Adv)'
                        elif 'Verb' in pos1 and 'Noun' in pos2:
                            pattern = 'Phrasal Verb (Verb + Noun)'
                        elif 'Noun' in pos1 and 'Verb' in pos2:
                            pattern = 'Phrasal Verb (Noun + Verb)'

                    if pattern:
                        # 1. Get the actual dictionary definition
                        original_def = synset.definition()
                        
                        # 2. Generate the formatted meaning
                        generated_def = generate_formatted_meaning(w1, w2, pattern)

                        results.append({
                            'Phrase': phrase.replace('_', ' '),
                            'Pattern': pattern,
                            'Generated_Meaning': generated_def,
                            'Original_Definition': original_def
                        })
                        seen_phrases.add(phrase)

    return pd.DataFrame(results)

# --- Step 4: Run and Save ---
df = extract_all_combinations()

# Remove empty entries if any
df = df[df['Generated_Meaning'] != ""]

# Save to CSV
output_file = 'WordNet_Complete_Combinations.csv'
df.to_csv(output_file, index=False)

print(f"\nSuccess! Processed {len(df)} combinations.")
print(f"Saved to: {output_file}")
print(df.head(10))