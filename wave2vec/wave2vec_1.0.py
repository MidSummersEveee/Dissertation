from datasets import load_dataset, load_metric

# timit = load_dataset("timit_asr")


import hub
timit = hub.load("hub://activeloop/timit-train")
# timit = hub.load("hub://activeloop/timit-test")

timit.summary()

# for i in range(10):
# 	print(timit['texts'][i:])

seq = timit.texts[0:10].numpy()
print(seq[0])