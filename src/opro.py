import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from src.LLMs import CustomLLM
from langchain.chat_models import ChatOpenAI

def create_chain(template, input_variables, temperature=0.5, callbacks=[], verbose=False, llm="Llama3"):
    if llm=="Llama3":
        prompt = PromptTemplate(input_variables=input_variables,template=template)
        chain = LLMChain(llm=CustomLLM(temperature=temperature),prompt=prompt)
    
    if llm=="GPT-4":
        prompt = PromptTemplate(input_variables=input_variables,template=template)
        chain = LLMChain(llm=ChatOpenAI(model_name="gpt-4", temperature=temperature, max_tokens=1000), prompt=prompt,callbacks=callbacks,verbose=verbose)
    
    if llm=="GPT-3.5":
        prompt = PromptTemplate(input_variables=input_variables, template=template)
        chain = LLMChain(llm=OpenAI(temperature=temperature, max_tokens=1000), prompt=prompt, callbacks=callbacks, verbose=verbose)
    return chain

def build_prompt_score(performance_df):
    return ''.join([f"Instruction: <INS> {performance_df.iloc[i]['Instruction']}</INS>\nScore:{performance_df.iloc[i]['score']}\n" for i in range(len(performance_df))])

def sort_instructions(performance_df,num_scores):
    performance_df = performance_df.sort_values(by='score')
    if len(performance_df) > num_scores:
        performance_df = performance_df.tail(num_scores)
    return performance_df

def sample_exemplars(df_train):
    return ''.join([f'''Instruction: <INS>\n\nStatement: {df_train.iloc[i]['claim']}\nLabel: {df_train.iloc[i]['final_label']}\n##############################################\n\n''' for i in range(len(df_train))])

def generate_prompts(optimizer_chain,texts_and_scores,exemplars,n_prompts=1):
    return [optimizer_chain.predict(texts_and_scores=texts_and_scores,exemplars=exemplars).replace("[","").replace("]","") for i in range(n_prompts)]

def are_the_same(x:list, y:list):
  results=[]
  for i in range(len(x)):
    if x[i]==y[i]:
      results.append(1)
    else:
      results.append(0)
  return results

def extract_label(sample_answer):
   '''
   TODO : You should complete this function for extracting your label.
   '''
   pass

def score_prompts(scorer_chain,prompts,eval_examples,performance_df):
  for prompt in prompts:
      pred_labels = []
      for index, example in eval_examples.iterrows():
          question = example['claim'] # Your text data (example: Tweet)
          answer = example['final_label'] # Your label (example: Positive)
          sample_answer = scorer_chain.predict(question=question,instruction=prompt)
          pred_labels.append(extract_label(sample_answer))
      scores = are_the_same(eval_examples['final_label'].values.tolist(), pred_labels)
      score=round(sum(scores)/len(scores), 2)
      performance_df = performance_df._append({'Instruction':prompt,'score':score},ignore_index=True)
  return performance_df

def opro(optimizer_chain, scorer_chain, performance_df, df_train, df_eval,n_scores=10, n_prompts=2, max_iterations=1):
    performance_df = sort_instructions(performance_df,n_scores)
    for _ in range(max_iterations):
        texts_and_scores = build_prompt_score(performance_df)
        train_examples = sample_exemplars(df_train)
        prompts = generate_prompts(optimizer_chain,texts_and_scores,train_examples,n_prompts)
        prompts = [x.split("<INS>")[1] for x in prompts]
        prompts = [x.split("</INS>")[0] for x in prompts]
        eval_examples = df_eval
        performance_df = score_prompts(scorer_chain,prompts,eval_examples,performance_df)
        performance_df = sort_instructions(performance_df,n_scores)
    return performance_df