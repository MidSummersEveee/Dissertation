from transformers import Wav2Vec2FeatureExtractor


feature_extractor = Wav2Vec2FeatureExtractor(
	feature_size=1,
	sampling_rate=16000,			# The sampling rate at which the model is trained on
	padding_value=0.0,				# shorter inputs need to be padded
	do_normalize=True,				# zero-mean-unit-variance normalized
	return_attention_mask=False)	
