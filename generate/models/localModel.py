from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LocalModel():

    def init(self, model_name, device_map):
        print('hello')
        self.model_name = model_name
        self.device_map = device_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device_map)

    def format_and_tokenize(self, messages, truncation, max_lenght, padding, add_generation_prompt):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        formatted_prompt = self.tokenizer.apply_chat_template(
                                                       conversation=messages,
                                                       add_generation_prompt=add_generation_prompt,
                                                       max_length=max_lenght,
                                                       truncation=truncation,
                                                       tokenize=False
                                                       )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=padding)
        input_ids = inputs['input_ids']
        input_ids = input_ids.to(self.model.device)
        print('input_ids dev', input_ids.device)
        attention_mask = inputs['attention_mask']
        attention_mask = attention_mask.to(self.model.device)
        print('attn_mask dev', attention_mask.device)

        return input_ids, attention_mask

    def generate(self, messages, temperature, max_new_tokens, truncation, max_lenght, padding, add_generation_prompt):

        input_ids, attention_mask = self.format_and_tokenize(messages, truncation, max_lenght, padding, add_generation_prompt)

        outputs = self.model.generate( input_ids=input_ids,
                                      attention_mask = attention_mask,
                                      max_new_tokens = max_new_tokens,
                                      temperature = temperature,
                                      pad_token_id=self.tokenizer.eos_token_id
                                       )
        response = outputs[0][input_ids.shape[-1]:]
        answer = self.tokenizer.decode(response, skip_special_tokens=True)
        return answer