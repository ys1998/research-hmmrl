import os, sys

def main():
	MAPPING = {}
	data_dir = sys.argv[1]
	for split in ['train', 'valid', 'test']:
		filepath = os.path.join(data_dir, split + '.txt')
		print("Mapping %s ..." % filepath)
		with open(filepath, 'r') as f:
			text = list(f.read().replace('\n', ' '))
		for _, word in enumerate(text):
			if word not in MAPPING:
				MAPPING[word] = 1
			else:
				MAPPING[word] += 1
	with open(os.path.join(data_dir, 'vocab'), 'w') as f:
		for c in sorted(MAPPING.keys()):
			f.write(c + '\n') 
		
if __name__=='__main__':
	main()