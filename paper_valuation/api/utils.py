import re
def merge_multi_page_result(all_pages_list):
    """
    Merges multiple 'answers' dictionaries. 
    If a question appears on multiple pages, the text is joined.
    """
    merged_answers = {}
    
    for page in all_pages_list:
        # Access the 'answers' dict from the result structure
        answers = page.get('answers', {})
        
        for q_label, text in answers.items():
            if q_label in merged_answers:
                # Add a space and append text for continuations
                merged_answers[q_label] += " " + text
            else:
                merged_answers[q_label] = text
    
    # Sort the dictionary by question number (Q1, Q2, etc.)
    sorted_answers = dict(sorted(
        merged_answers.items(),
        key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0
    ))
    
    return {"answers": sorted_answers, "total_pages": len(all_pages_list)}