import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.output_parsers import PydanticOutputParser
from Prompts.prompt import appointment_prompt
from Configurations.config import llm_model
from PydanticModels.model import UserSymptoms, PossibleCauses



parser = PydanticOutputParser(pydantic_object=PossibleCauses)



# Chain it all together
def get_possible_causes(user_input: UserSymptoms):
    formatted_prompt = appointment_prompt.format(
        symptoms=user_input.symptoms,
        description=user_input.user_description,
        format_instructions=parser.get_format_instructions()
    )

    response = llm_model.LLM().invoke(formatted_prompt)
    return parser.parse(response.content)


# # Example usage
# if __name__ == "__main__":
#     user_input = UserSymptoms(
#         symptoms=["chest pain", "shortness of breath", "fatigue"],
#         user_description="I've been having chest tightness for the past few hours, and it's getting worse when I walk."
#     )

#     result = get_possible_causes(user_input)
#     print(result)
