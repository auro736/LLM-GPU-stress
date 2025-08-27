from openai import OpenAI
from models.togetherModel import TogetherModel

class BaseAgent():

    def __init__(self, model_type, model_name, api_key):
        self.model_type = model_type

        if self.model_type == 'together':
            self.model = TogetherModel(model_name=model_name, api_key=api_key)
        elif self.model_type == 'openai':
            self.model = OpenAI(api_key=api_key)
            self.model_name = model_name
    
    def replace_system_with_developer(self,messages):
        new_messages = []
        for m in messages:
            role = m.get('role')
            if isinstance(role, str) and role.strip().lower() == 'system':
                new_messages.append({**m, 'role': 'developer'})
            else:
                new_messages.append(dict(m))  # copia superficiale
        return new_messages

    
    def generate(self, messages, temperature, max_new_tokens, seed):
        if self.model_type == 'together':
            answer = self.model.generate(messages=messages, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
        elif self.model_type == 'openai':
            messages = self.replace_system_with_developer(messages)
            completion = self.model.chat.completions.create(
                model=self.model_name,
                messages=messages,
                seed=seed,
                temperature=temperature,
                max_completion_tokens=max_new_tokens
            )
            answer = completion.choices[0].message
        return answer
        

    

    