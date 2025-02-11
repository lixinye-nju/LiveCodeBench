import json

try:
    from anthropic import HUMAN_PROMPT, AI_PROMPT
except ImportError:
    HUMAN_PROMPT = None
    AI_PROMPT = None

from enum import Enum
from lcb_runner.lm_styles import LMStyle
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem


class Language(Enum):
    PYTHON = "python"
    CPP = "cpp"
    JAVA = "java"
    
    
COMMENT_SYMBOL = {
    Language.PYTHON: "#",
    Language.CPP: "//",
    Language.JAVA: "//"
}


class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = {
        Language.PYTHON: "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.",
        Language.CPP: "You are an expert C++ programmer. You will be given a question (problem specification) and will generate a correct C++ program that matches the specification and passes all tests.",
        Language.JAVA: "You are an expert Java programmer. You will be given a question (problem specification) and will generate a correct Java program that matches the specification and passes all tests."
    }

    SYSTEM_MESSAGE_GEMINI = {
        Language.PYTHON: "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution.",
        Language.CPP: "You are an expert C++ programmer. You will be given a question (problem specification) and will generate a correct C++ program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution.",
        Language.JAVA: "You are an expert Java programmer. You will be given a question (problem specification) and will generate a correct Java program that matches the specification and passes all tests. Do NOT use system calls like `exit` in the generated program. Ensure that the first code block contains the solution."
    }

    SYSTEM_MESSAGE_GEMINITHINK = {
        Language.PYTHON: "You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests.",
        Language.CPP: "You are an expert C++ programmer. You will be given a question (problem specification) and will generate a correct C++ program that matches the specification and passes all tests.",
        Language.JAVA: "You are an expert Java programmer. You will be given a question (problem specification) and will generate a correct Java program that matches the specification and passes all tests."
    }

    SYSTEM_MESSAGE_DEEPSEEK = {
        Language.PYTHON: "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science.",
        Language.CPP: "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science.", 
        Language.JAVA: "You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science." 
    }

    SYSTEM_MESSAGE_CODEQWEN = {
        Language.PYTHON: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user",
        Language.CPP: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user",
        Language.JAVA: "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    }

    FORMATTING_MESSAGE_WITH_STARTER_CODE = {
        Language.PYTHON: "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.",
        Language.CPP: "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.",
        Language.JAVA: "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    }

    FORMATTING_WITHOUT_STARTER_CODE = {
        Language.PYTHON: "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the Python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.",
        Language.CPP: "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the C++ program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.",
        Language.JAVA: "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the Java program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.",
    }


def get_generic_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE[language]}\n"
        )
        prompt += f"```{language}\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE[language]}\n"
        prompt += f"```{language}\n{COMMENT_SYMBOL[language]} YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_oaireason_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE[language]}\n"
        )
        prompt += f"```{language}\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: Implement a function called `main()` which orchastrates the solution by reading inputs from stdin and writing the answer to stdout. Feel free to use additional functions as necessary. Next do NOT forget to call `main` function at the end of the program otherwise you will not be awarded any points.\n"
        prompt += f"```{language}\n{COMMENT_SYMBOL[language]} YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_geminithinking_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE[language]}\n"
        )
        prompt += f"```{language}\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE[language]}\n"
        prompt += f"```{language}\n{COMMENT_SYMBOL[language]} YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def get_deepseekcode_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    prompt = f"### Instruction: You will be given a question (problem specification) and will generate a correct {language} program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Instruction: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE[language]}\n"
        )
        prompt += f"```{language}\n{question.starter_code}\n```\n\n"
    else:
        prompt += (
            f"### Instruction: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE[language]}\n"
        )
        prompt += f"```{language}\n{COMMENT_SYMBOL[language]} YOUR CODE HERE\n```\n\n"
    prompt += f"### Response:\n\n"
    return prompt


def get_qwen_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "/abacus/models/Qwen1.5-72B-Chat/", padding_side="left", use_fast=False
    )
    prompt = f"You will be given a question (problem specification) and will generate a correct {language} program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE[language]}\n"
        prompt += f"```{language}\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE[language]}\n\n"
        prompt += f"```{language}\n{COMMENT_SYMBOL[language]} YOUR CODE HERE\n```\n\n"

    messages = [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC[language]},
        {"role": "user", "content": prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        truncation=False,
        padding=False,
    )
    return prompt


def get_codeqwen_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    prompt = f"You will be given a question (problem specification) and will generate a correct {language} program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
    prompt += f"Question: {question.question_content}\n\n"
    if question.starter_code:
        prompt += f"{PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE[language]}\n"
        prompt += f"```{language}\n{question.starter_code}\n```\n\n<|im_end|>\n"
    else:
        prompt += f"{PromptConstants.FORMATTING_WITHOUT_STARTER_CODE[language]}\n"
        prompt += f"```{language}\n# YOUR CODE HERE\n```\n\n<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    return prompt


with open("lcb_runner/prompts/few_shot_examples/generation/func.json") as f:
    func = json.load(f)

with open("lcb_runner/prompts/few_shot_examples/generation/stdin.json") as f:
    stdin = json.load(f)


def get_base_model_question_template_answer(question: CodeGenerationProblem, language: Language = Language.PYTHON):
    if question.starter_code:
        examples_json = func
    else:
        examples_json = stdin

    def get_example_prompt(example):
        prompt = ""
        prompt += "### Question\n"
        prompt += example["question"]
        prompt += "\n\n"
        if question.starter_code:
            prompt += "### Starter Code\n"
            prompt += example["sample_code"]
            prompt += "\n\n"
        prompt += "### Answer\n\n"
        prompt += example["answer"]
        if example["answer"]:
            prompt += "\n\n"
        return prompt

    prompt = ""
    prompt += get_example_prompt(examples_json[0])
    prompt += get_example_prompt(
        {
            "question": question.question_content,
            "sample_code": question.starter_code,
            "answer": "",
        }
    )
    return prompt


def format_prompt_generation(
    question: CodeGenerationProblem, LanguageModelStyle: LMStyle, language: Language.PYTHON
) -> str:
    if LanguageModelStyle in [LMStyle.OpenAIChat, LMStyle.DeepSeekAPI]:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC[language],
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, language),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.OpenAIReasonPreview:
        chat_messages = [
            {
                "role": "user",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC[language]
                + "\n\n"
                + get_generic_question_template_answer(question[language]),
            },
        ]
        return chat_messages
    elif LanguageModelStyle == LMStyle.OpenAIReason:
        chat_messages = [
            {
                "role": "user",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC[language]
                + "\n\n"
                + get_oaireason_question_template_answer(question, language),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.LLaMa3:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC[language],
            },
        ]
        chat_messages += [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, language),
            },
        ]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct", padding_side="left", use_fast=False
        )
        return tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
            truncation=False,
            padding=False,
        )

    if LanguageModelStyle == LMStyle.Claude:
        prompt = f"{HUMAN_PROMPT}\n"
        prompt += f"{PromptConstants.SYSTEM_MESSAGE_GENERIC[language]}\n\n"
        prompt += f"{get_generic_question_template_answer(question, language).rstrip()}\n"
        prompt += f"{AI_PROMPT}"
        return prompt

    if LanguageModelStyle == LMStyle.Claude3:
        system = PromptConstants.SYSTEM_MESSAGE_GENERIC[language]
        prompt = [
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, language).rstrip(),
            }
        ]
        return system, prompt

    if LanguageModelStyle == LMStyle.Gemini:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GEMINI[language]}\n"
        prompt += f"{get_generic_question_template_answer(question, language)}"
        return prompt

    if LanguageModelStyle == LMStyle.GeminiThinking:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_GEMINITHINK[language]}\n"
        prompt += f"{get_geminithinking_question_template_answer(question, language)}"
        return prompt

    if LanguageModelStyle == LMStyle.MistralWeb:
        chat_messages = [
            {
                "role": "system",
                "content": PromptConstants.SYSTEM_MESSAGE_GENERIC[language],
            },
            {
                "role": "user",
                "content": get_generic_question_template_answer(question, language),
            },
        ]
        return chat_messages

    if LanguageModelStyle == LMStyle.DeepSeekCodeInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_DEEPSEEK[language]}\n\n"
        prompt += f"{get_deepseekcode_question_template_answer(question, language)}"
        return prompt

    if LanguageModelStyle == LMStyle.CodeQwenInstruct:
        prompt = f"{PromptConstants.SYSTEM_MESSAGE_CODEQWEN[language]}\n\n"
        prompt += f"{get_codeqwen_question_template_answer(question, language)}"
        return prompt

    if LanguageModelStyle == LMStyle.GenericBase:
        if language != Language.PYTHON:
            raise NotImplementedError(f"Language {language} for base model is not implemented.")
        prompt = get_base_model_question_template_answer(question)
        return prompt

    raise NotImplementedError(
        f"LanguageModelStyle {LanguageModelStyle} not implemented"
    )


def test():
    import pathlib

    base_dir = "logs/example_prompts/generation"
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)

    for lmstyle in LMStyle:
        generation_problem = CodeGenerationProblem(
            "title",
            "question-content",
            "leetcode",
            "question_id",
            "contest_id",
            "contest_date",
            "",
            "easy",
            "[]",
            "[]",
            "{}",
        )
        prompt1 = format_prompt_generation(generation_problem, lmstyle)
        with open(f"{base_dir}/{lmstyle}_1.txt", "w") as f:
            try:
                f.write(prompt1)
            except TypeError:
                f.write(json.dumps(prompt1))

        generation_problem.starter_code = "starter code"
        prompt2 = format_prompt_generation(generation_problem, lmstyle)
        with open(f"{base_dir}/{lmstyle}_2.txt", "w") as f:
            try:
                f.write(prompt2)
            except TypeError:
                f.write(json.dumps(prompt2))


if __name__ == "__main__":
    test()
