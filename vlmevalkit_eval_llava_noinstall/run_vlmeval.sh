python run.py --data HallusionBench MME POPE OCRVQA_TESTCORE SEEDBench_IMG --model llava_v1.5_7b --verbose --work-dir output --alphavalue 5 --rvalue 16
# rvalue 12: 50% reduction
# rvalue 16: 66% reduction
# rvalue 18: 75% reduction
# the reduction ratio is calculated by: rvalue*num_layers/original_seq_len, for LLaVA1.5, original_seq_len=576, num_layers=24
# alphavalue: set from 0-5, hyperparameter for matching function