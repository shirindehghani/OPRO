import pandas as pd
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import re

def create_chain_from_template(template, input_variables, temperature=.3,callbacks=[], verbose=True):
    prompt = PromptTemplate(
        input_variables=input_variables,
        template=template)
    chain = LLMChain(
        llm=OpenAI(temperature=temperature, max_tokens=1000),
        prompt=prompt,callbacks=callbacks,verbose=verbose)
    return chain

def instruction_score_pair(df):
    return ''.join([f"<INS>:{df.iloc[i]['text']}</INS>\nScore:{df.iloc[i]['score']}\n" for i in range(len(df))])

def rank_instructions(df,num_scores):
    df = df.sort_values(by='score')
    if len(df) > num_scores:
        df = df.tail(num_scores)
    return df

def sample_exemplars(df_train):
    return ''.join([f'''Input:<INS>\nQ:Which of three mentioned labels is the best fit the following tweet? Answer with only the previous options that are most accurate and nothing else. Just name one of them with no more explanation.\nTweet: {df_train.iloc[i]['full_text']}\noutput:{df_train.iloc[i]['3signals']}\n\n\n''' for i in range(len(df_train))])

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
  sample_answer=sample_answer.lower()
  label=-1
  if ("bullish" in sample_answer) or ("positive" in sample_answer):
    label="Bullish"
  if ("bearish" in sample_answer) or ("negative" in sample_answer):
    label="Bearish"
  if ("neither" in sample_answer) or ("neutral" in sample_answer):
    label="Neither"
  return label

def score_prompts(scorer_chain,prompts,eval_examples,performance_df):
  for prompt in prompts:
      pred_labels = []
      for index, example in eval_examples.iterrows():
          question = example['full_text']
          answer = example['3signals']
          sample_answer = scorer_chain.predict(question=question,instruction=prompt)
          pred_labels.append(extract_label(sample_answer))
      scores = are_the_same(eval_examples['3signals'].values.tolist(), pred_labels)
      score=round(sum(scores)/len(scores), 2)
      print("+++++++++++++++++++++++++Score: ", score)
      # score=f1_score(eval_examples['3signals'].values.tolist(),
      #                pred_labels, average="macro")
      performance_df = performance_df._append({'text':prompt,'score':score},ignore_index=True)
  return performance_df

def opro(optimizer_chain, scorer_chain, performance_df, df_train, df_eval,n_scores=10, n_prompts=2, max_iterations=1):
    performance_df = rank_instructions(performance_df,n_scores)
    for _ in range(max_iterations):
        texts_and_scores = build_text_and_scores(performance_df)
        train_examples = sample_exemplars(df_train)
        prompts = generate_prompts(optimizer_chain,texts_and_scores,train_examples,n_prompts)
        eval_examples = df_eval
        performance_df = score_prompts(scorer_chain,prompts,eval_examples,performance_df)
        performance_df = rank_instructions(performance_df,n_scores)
    return performance_df

if __name__ == '__main__':
    optimizer_chain = create_chain_from_template(meta_prompt,["texts_and_scores","exemplars"],verbose=True)
    scorer_chain = create_chain_from_template(scorer_prompt,["question","instruction"],verbose=True)

    performance_df = performance_data
    df_train=data_train
    df_eval=data_eval
    output = opro(optimizer_chain,scorer_chain,performance_df,df_train, df_eval,n_scores=2,n_prompts=1,max_iterations=1)
    # print(output)
    # output.to_csv("data/performance2.csv")