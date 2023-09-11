from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True)

# gpu
# model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).half().cuda()

# cpu
model = AutoModel.from_pretrained("THUDM/chatglm2-6b-int4", trust_remote_code=True).half().float()

model = model.eval()

response, history = model.chat(tokenizer, "你好", history=[])

print(response)

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)

print(response)
