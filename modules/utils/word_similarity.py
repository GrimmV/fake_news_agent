from difflib import SequenceMatcher

def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def find_most_similar_word(string_list: list[str], string_in):
    
    tmp_score = 0
    most_similar_str = ""
    
    for elem in string_list:
        score = similar(string_in, elem)
        if score > tmp_score:
            tmp_score = score
            most_similar_str = elem
            
    return most_similar_str