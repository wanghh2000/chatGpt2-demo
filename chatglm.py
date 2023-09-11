#!/usr/bin/env python
# _*_coding:utf-8_*_
# Author   :    Junhui Yu

import re
import json
from transformers import AutoTokenizer, AutoModel

# 根据需求补充
schema = {
    'JD岗位要求': ['学历要求', '专业要求', '工作年限要求', '编程语言', '加分项']
}

IE_PATTERN = "{}\n\n提取上述句子中{}类型的实体，输出JSON格式，上述句子中不存在的信息用['该JD未要求']来表示，多个值之间用','分隔。"

ie_examples = {
    'JD岗位要求':
        {
            'sentence': '职位要求：1、硕士以上学历。2、计算机相关专业。3、3年以上工作经验。4、熟练掌握python或者c++语言。5、有自然语言处理获奖经历优先',
            'answers': {
                '学历要求': ['硕士'],
                '专业要求': ['计算机'],
                '工作年限要求': ['3年以上'],
                '编程语言': ['python', 'c++'],
                '加分项': ['自然语言处理获奖经历'],
            }
        }
}


def init_prompts():
    ie_prefix = [
        (
            "需要你协助完成信息抽取任务，当我给你一个JD职位要求时，帮我抽取出句子中三元组，并按照JSON的格式输出，上述句子中没有的信息用['该JD未要求']来表示，多个值之间用','分隔。",
            '请输入JD职位描述。'
        )
    ]
    properties_str = ', '.join(schema['JD岗位要求'])
    schema_str_list = f'“JD岗位要求”({properties_str})'
    sentence = ie_examples['JD岗位要求']['sentence']
    sentence_with_prompt = IE_PATTERN.format(sentence, schema_str_list)
    ie_prefix.append((
        f'{sentence_with_prompt}',
        f"{json.dumps(ie_examples['JD岗位要求']['answers'], ensure_ascii=False)}"
    ))

    return {'ie_prefix': ie_prefix}


def format_output(response: str):
    if '```json' in response:
        res = re.findall(r'```json(.*?)```', response)
        if len(res) and res[0]:
            response = res[0]
        response.replace('、', ',')
    try:
        return json.loads(response)
    except:
        return response


def inference(sentences: list, custom_settings: dict):
    for sentence in sentences:
        properties_str = ', '.join(schema['JD岗位要求'])
        schema_str_list = f'“JD岗位要求”({properties_str})'
        sentence_with_ie_prompt = IE_PATTERN.format(sentence, schema_str_list)
        result, _ = model.chat(tokenizer, sentence_with_ie_prompt, history=custom_settings['ie_prefix'])
        result = format_output(result)
        print(sentence)
        print(result)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()

    sentences = [
        '职位要求：1、本科以上学历。2、电子信息或软件工程专业。3、1-3年工作经验。4、熟练掌握java或者c++语言。5、有相关项目经验优先',
    ]

    custom_settings = init_prompts()
    inference(sentences,
        custom_settings
    )

