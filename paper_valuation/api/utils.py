import re
import re

import re

def merge_multi_page_result(all_pages_list):
    """
    Merges multiple 'answers' dictionaries across pages.
    
    NEW LOGIC:
    - If a question label appears on multiple pages (e.g., Q2 on page 1 and Q2 on page 2),
      the answers are concatenated with a space.
    - If UNLABELED_CONTINUATION exists, it's appended to the last question found.
    
    Example:
    Page 1: Q1 (complete), Q2 (partial)
    Page 2: Q2 (continuation - student wrote "Q2" again), Q3 (complete)
    
    Result: Q1 (full), Q2 (page1 + page2), Q3 (full)
    """
    merged_answers = {}
    last_q_label = None
    
    for page_index, page in enumerate(all_pages_list):
        # Access the 'answers' dict from the result structure
        answers = page.get('answers', {})
        
        # 1. Handle UNLABELED_CONTINUATION (student forgot to write question number)
        if 'UNLABELED_CONTINUATION' in answers:
            unlabeled_text = answers.pop('UNLABELED_CONTINUATION')
            
            if last_q_label and last_q_label in merged_answers:
                # Append to the last question from previous page
                merged_answers[last_q_label] += " " + unlabeled_text.strip()
                print(f"⚠️  Page {page_index + 1}: Unlabeled continuation appended to {last_q_label}")
                print(f"    (Student should have written '{last_q_label}' at the top of this page)")
            else:
                # Edge case: First page is unlabeled (very unusual)
                if 'Q1' in merged_answers:
                    merged_answers['Q1'] = unlabeled_text.strip() + " " + merged_answers['Q1']
                else:
                    merged_answers['Q1'] = unlabeled_text.strip()
                last_q_label = 'Q1'
                print(f"⚠️  Page {page_index + 1}: Unlabeled page assigned to Q1 by default")

        # 2. Process labeled questions on this page
        for q_label, text in answers.items():
            if q_label in merged_answers:
                # Question label appears again (continuation page)
                merged_answers[q_label] += " " + text.strip()
                print(f"✅ Page {page_index + 1}: {q_label} continuation merged (student correctly labeled it)")
            else:
                # New question label
                merged_answers[q_label] = text.strip()
                print(f"✅ Page {page_index + 1}: Added {q_label}")
            
            # Update the last label found for next page's unlabeled continuation
            last_q_label = q_label
    
    # Final step: Sort the dictionary by question number (Q1, Q2, Q3, ...)
    sorted_answers = dict(sorted(
        merged_answers.items(),
        key=lambda x: int(re.search(r'\d+', x[0]).group()) if re.search(r'\d+', x[0]) else 0
    ))
    
    print(f"\n📊 Final Merge Summary: {len(sorted_answers)} questions across {len(all_pages_list)} pages")
    
    return {"answers": sorted_answers, "total_pages": len(all_pages_list)}