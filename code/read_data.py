import random

def check_fasta_file(file_path):
    if not os.path.isfile(file_path):
        raise Exception(f"Error: \'{file_path}\' does not exist")
    
    # 检查文件扩展名是否是 .fasta 或 .fa
    if not file_path.endswith(('.fasta', '.fa')):
        raise Exception(f"'{file_path}'is not a fasta file")

def check_for_unstandard_amino_acids(sequence):
    standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    for amino_acid in sequence:
        if amino_acid not in standard_amino_acids:
            return False
    return True

def read_data(pos_file,neg_file,train_num,test_num,seed):
    P_File = open(pos_file,'r')
    P_Sequences = []
    for line in P_File.readlines():
        if not line.startswith('>'):
            sequence = line.rstrip()
            if check_for_unstandard_amino_acid(sequence):
                P_Sequences.append([1,sequence])
    P_File.close()
    random.seed(seed)
    random.shuffle(P_Sequences)
    N_File = open(neg_file,'r')
    N_Sequences = []
    for line in N_File.readlines():
        if not line.startswith('>'):
            sequence = line.rstrip()
            if check_for_unstandard_amino_acid(sequence):
                N_Sequences.append([0,sequence])
    N_File.close()
    random.seed(seed)
    random.shuffle(N_Sequences)
    if train_num+test_num > len(P_Sequences):
        raise Exception("The sum train_num and test_num is greater than the number of positive samples. Note that sequences containing non-standard amino acids are excluded.")
    if train_num+test_num > len(N_Sequences):
        raise Exception("The sum train_num and test_num is greater than the number of negative samples. Note that sequences containing non-standard amino acids are excluded.")
    train_data = P_Sequences[:train_num] + N_Sequences[:train_num]
    test_data = P_Sequences[-1*test_num:] + N_Sequences[-1*test_num:]
    return train_data,test_data
