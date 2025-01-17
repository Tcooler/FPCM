import random
def read_data(pos_file,neg_file,seed):
    P_File = open(pos_file,'r')
    P_Sequences = []
    for line in P_File.readlines():
        if not line.startswith('>'):
            sequence = line.rstrip()
            P_Sequences.append([1,sequence])
    P_File.close()
    random.seed(seed)
    random.shuffle(P_Sequences)
    N_File = open(neg_file,'r')
    N_Sequences = []
    for line in N_File.readlines():
        if not line.startswith('>'):
            sequence = line.rstrip()
            N_Sequences.append([0,sequence])
    N_File.close()
    random.seed(seed)
    random.shuffle(N_Sequences)
    train_data = P_Sequences[:20] + N_Sequences[:20]
    test_data = P_Sequences[20:] + N_Sequences[20:]
    return train_data,test_data
