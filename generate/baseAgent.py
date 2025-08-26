from models.togetherModel import TogetherModel

class BaseAgent():

    def __init__(self, model_type, model_name, api_key):
        self.model_type = model_type

        if self.model_type == 'together':
            self.model = TogetherModel(model_name=model_name, api_key=api_key)
        elif self.model_type == 'openai':
            pass
    
    def generate(self, messages, temperature, max_new_tokens, seed):
        if self.model_type == 'together':
            answer = self.model.generate(messages=messages, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
        elif self.model_type == 'openai':
            pass
        return answer
        

    

    