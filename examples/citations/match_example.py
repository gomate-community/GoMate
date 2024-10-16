from gomate.modules.citation.match_citation import MatchCitation
import json

mc = MatchCitation()

with open(f'sample5.json', 'r', encoding='utf-8') as f:
    input_data = json.load(f)
print(input_data)
result = mc.ground_response(
    question=input_data["question"],
    response=input_data["response"],
    evidences=input_data["evidences"],
    selected_idx=input_data["selected_idx"],
    markdown=True,
    show_code=True,
    selected_docs=input_data["selected_docs"],
)

print(result)
