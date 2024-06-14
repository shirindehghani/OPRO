meta_prompt = """Your task is to generate the instruction <INS>. Below are some previous instructions with their scores. The score ranges from 0 to 1.

{texts_and_scores}

Some examples of template variable inputs and expected outputs are given below to illustrate the task.

{exemplars}

Generate an instruction that is different from all the instructions <INS> above, and has a higher score
than all the instructions <INS> above. The instruction should begin with <INS> and end with </INS>.
The instruction should be concise, effective, and generally applicable to all problems above.
"""

scorer_prompt = """
{instruction}

Statement: {question}
Label:
"""